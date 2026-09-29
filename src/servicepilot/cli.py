from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import ServiceAgent
from .context_manager import ContextCompactor, DeepSeekContextSummaryModel
from .graph_runtime import LangGraphServiceRuntime
from .memory import (
    FallbackSessionStore,
    InMemorySessionStore,
    RedisSessionStore,
    SQLiteSessionStore,
)
from .observability import TraceService, current_trace, event
from .planning import ChatCompletionPlannerModel, TaskPlanner
from .providers import (
    APIEmbeddingEncoder,
    OnlineModelSettings,
    ResponsesIntentModel,
    ResponsesPolicyReranker,
    UrllibJsonTransport,
)
from .repository import SQLiteCommerceRepository
from .retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    ChromaPolicyRetriever,
    FastEmbedTextEncoder,
    HashingTextEncoder,
    HybridPolicyRetriever,
    load_policy_chunks,
)
from .routing import (
    CachedVotingIntentRouter,
    EmbeddingIntentRouter,
    LLMIntentRouter,
    RuleIntentRouter,
)


def build_agent(
    data_dir: Path,
    *,
    session_db: Path | None = None,
    online: bool = False,
    use_graph: bool = True,
    chroma_dir: Path | None = None,
    redis_url: str | None = None,
    commerce_db: Path | None = None,
    dynamic: bool | None = None,
    planner=None,
    trace_service=None,
    failure_memory=None,
    experience_extractor=None,
    naive_reflections=None,
) -> ServiceAgent | LangGraphServiceRuntime:
    database = commerce_db or os.getenv("SERVICEPILOT_COMMERCE_DB") or session_db or ":memory:"
    repository = SQLiteCommerceRepository.from_json(data_dir / "orders.json", database=database)
    if dynamic is None:
        dynamic = os.getenv("SERVICEPILOT_DYNAMIC_PLANNING", "0").lower() in {"1", "true", "yes"}
    if dynamic and not use_graph:
        raise ValueError("动态规划需要 LangGraph；请关闭 legacy-runtime")
    if session_db is None and database != ":memory:":
        session_db = Path(database)
    store = SQLiteSessionStore(session_db) if session_db else None
    if redis_url:
        store = FallbackSessionStore(
            RedisSessionStore.from_url(redis_url),
            store or InMemorySessionStore(),
        )
    chunks = load_policy_chunks(data_dir / "policies.json")
    intent_model = None
    if online:
        settings = OnlineModelSettings.from_env()
        transport = UrllibJsonTransport(settings)
        if settings.embedding_mode == "api":
            encoder = APIEmbeddingEncoder(transport, settings.embedding_model)
        elif settings.embedding_mode == "local":
            encoder = FastEmbedTextEncoder(settings.embedding_model)
        else:
            encoder = HashingTextEncoder()
        vector = (
            ChromaPolicyRetriever(chunks, encoder, persist_directory=chroma_dir)
            if chroma_dir
            else HybridPolicyRetriever.from_json(data_dir / "policies.json", encoder=encoder).vector
        )
        intent_model = ResponsesIntentModel(transport, settings.intent_model)
        router = CachedVotingIntentRouter(
            (
                (
                    "llm",
                    LLMIntentRouter(intent_model),
                    0.5,
                ),
                ("embedding", EmbeddingIntentRouter(encoder), 0.2),
                ("keyword", RuleIntentRouter(), 0.3),
            )
        )
    else:
        encoder = HashingTextEncoder()
        vector = (
            ChromaPolicyRetriever(chunks, encoder, persist_directory=chroma_dir)
            if chroma_dir
            else HybridPolicyRetriever.from_json(data_dir / "policies.json").vector
        )
        router = CachedVotingIntentRouter(
            (
                ("embedding", EmbeddingIntentRouter(encoder), 0.2),
                ("keyword", RuleIntentRouter(), 0.8),
            )
        )
    reranker = ResponsesPolicyReranker(transport, settings.intent_model) if online else None
    retriever = AdvancedHybridPolicyRetriever(
        BM25PolicyRetriever(chunks), vector, reranker=reranker
    )
    from .corrective_rag import configure_rag

    retriever = configure_rag(retriever, chunks, vector)
    compact_enabled = os.getenv("SERVICEPILOT_CONTEXT_COMPACTION", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    context_model = (
        DeepSeekContextSummaryModel(
            transport, os.getenv("SERVICEPILOT_CONTEXT_MODEL") or "deepseek-chat"
        )
        if compact_enabled and online
        else None
    )
    agent = ServiceAgent(
        repository,
        retriever,
        router=router,
        session_store=store,
        context_compactor=ContextCompactor.from_env(context_model) if compact_enabled else None,
    )
    agent.pdf_encoder = encoder
    agent.trace_service = trace_service or TraceService.create(
        os.getenv("SERVICEPILOT_TRACE_DB") or database
    )
    if dynamic and planner is None and os.getenv("SERVICEPILOT_PLANNER_MODEL"):
        settings = OnlineModelSettings.from_env()
        planner = TaskPlanner(
            ChatCompletionPlannerModel(
                UrllibJsonTransport(settings), os.environ["SERVICEPILOT_PLANNER_MODEL"]
            )
        )
    memory_mode = os.getenv("SERVICEPILOT_FAILURE_MEMORY_MODE", "").upper()
    if memory_mode and memory_mode not in {"B0", "B1", "B2"}:
        raise ValueError("SERVICEPILOT_FAILURE_MEMORY_MODE must be B0, B1 or B2")
    if dynamic and memory_mode == "B1" and naive_reflections is None:
        from .failure_memory import NaiveReflectionStore

        naive_reflections = NaiveReflectionStore(
            os.getenv("SERVICEPILOT_FAILURE_MEMORY_DB") or database
        )
    memory_enabled = memory_mode == "B2" or (
        not memory_mode and os.getenv("SERVICEPILOT_FAILURE_MEMORY", "0") == "1"
    )
    if dynamic and failure_memory is None and memory_enabled:
        from .failure_memory import ExperienceExtractor, FailureMemoryStore

        memory_db = os.getenv("SERVICEPILOT_FAILURE_MEMORY_DB") or database
        failure_memory = FailureMemoryStore(
            memory_db,
            encoder=encoder,
            chroma_directory=(chroma_dir / "failure_memory") if chroma_dir else None,
            capacity=int(os.getenv("SERVICEPILOT_FAILURE_MEMORY_CAPACITY", "100")),
        )
        if online and experience_extractor is None:
            experience_extractor = ExperienceExtractor(
                transport, os.getenv("SERVICEPILOT_FAILURE_MEMORY_MODEL") or settings.intent_model
            )
    if (
        intent_model is not None
        and failure_memory is not None
        and os.getenv("SERVICEPILOT_FAILURE_MEMORY_ROUTING", "0") == "1"
    ):
        from .failure_memory import FailureMemoryRetriever

        memory_retriever = FailureMemoryRetriever(failure_memory)

        def routing_lessons(_message):
            collector = current_trace()
            return memory_retriever.retrieve(
                goal="intent routing",
                task_type="intent_routing",
                stage="routing",
                conditions={"intent_routing"},
                trace_id=collector.record["trace_id"] if collector else None,
                task_id=collector.record.get("task_id") if collector else None,
            )

        def mark_routing_lesson(item):
            collector = current_trace()
            failure_memory.record_usage(
                item.memory_id,
                trace_id=collector.record["trace_id"] if collector else None,
                task_id=collector.record.get("task_id") if collector else None,
                stage="routing",
                score=item.score,
                applied=True,
            )
            event(
                "FAILURE_MEMORY_APPLIED",
                memory_id=item.memory_id,
                score=item.score,
                applicable_reason=item.why_relevant,
                stage="routing",
            )

        intent_model.lesson_provider = routing_lessons
        intent_model.on_lesson_injected = mark_routing_lesson
    return (
        LangGraphServiceRuntime(
            agent,
            dynamic=dynamic,
            planner=planner,
            failure_memory=failure_memory,
            experience_extractor=experience_extractor,
            naive_reflections=naive_reflections,
        )
        if use_graph
        else agent
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ServicePilot terminal demo")
    parser.add_argument("--user", default="USER-001")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument("--session-db", type=Path, help="SQLite path for durable sessions")
    parser.add_argument("--session-id", help="Resume a session stored in --session-db")
    parser.add_argument(
        "--online", action="store_true", help="Use configured LLM and Embedding APIs"
    )
    parser.add_argument("--legacy-runtime", action="store_true", help="Disable LangGraph workflow")
    parser.add_argument("--chroma-dir", type=Path, help="Persist the vector index in Chroma")
    parser.add_argument("--redis-url", help="Redis URL for TTL-backed short-term memory")
    parser.add_argument("--commerce-db", type=Path, help="Durable business ledger")
    parser.add_argument("--dynamic-planning", action="store_true", default=None)
    args = parser.parse_args()
    agent = build_agent(
        args.data_dir,
        session_db=args.session_db,
        online=args.online,
        use_graph=not args.legacy_runtime,
        chroma_dir=args.chroma_dir,
        redis_url=args.redis_url,
        commerce_db=args.commerce_db,
        dynamic=args.dynamic_planning,
    )
    session_id: str | None = args.session_id

    print("ServicePilot 已启动。输入 quit 退出。")
    while True:
        message = input("你：").strip()
        if message.lower() in {"quit", "exit"}:
            break
        response = agent.chat(user_id=args.user, message=message, session_id=session_id)
        session_id = response.session.session_id
        print(f"客服：{response.message}")
        print(f"会话：{session_id}")
        if response.citations:
            print(f"政策依据：{', '.join(response.citations)}")


if __name__ == "__main__":
    main()
