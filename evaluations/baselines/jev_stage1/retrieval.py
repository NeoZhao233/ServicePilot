from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import PolicyChunk, RetrievalHit
from .observability import Span, event, observed, submit_with_context

_LATIN_OR_NUMBER = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without external dependencies."""
    normalized = text.lower()
    latin = _LATIN_OR_NUMBER.findall(normalized)
    cjk_chars = _CJK.findall(normalized)
    cjk_bigrams = ["".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1)]
    return latin + cjk_chars + cjk_bigrams


class PolicySearch(Protocol):
    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]: ...


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class PolicyReranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalHit], *, limit: int
    ) -> list[RetrievalHit]: ...


def load_policy_chunks(path: str | Path) -> list[PolicyChunk]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        PolicyChunk(
            policy_id=item["policy_id"],
            title=item["title"],
            content=item["content"],
            categories=tuple(item.get("categories", [])),
            tags=tuple(item.get("tags", [])),
        )
        for item in payload
    ]


def chunk_text(chunk: PolicyChunk) -> str:
    return f"{chunk.title} {chunk.content} {' '.join(chunk.tags)}"


class BM25PolicyRetriever:
    """Small BM25 policy retriever used as the first reproducible baseline."""

    def __init__(self, chunks: list[PolicyChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._documents = [tokenize(chunk_text(chunk)) for chunk in chunks]
        self._frequencies = [Counter(document) for document in self._documents]
        self._average_length = (
            sum(map(len, self._documents)) / len(self._documents) if self._documents else 0.0
        )
        self._document_frequency = Counter(
            term for document in self._documents for term in set(document)
        )

    @classmethod
    def from_json(cls, path: str | Path) -> BM25PolicyRetriever:
        return cls(load_policy_chunks(path))

    @observed("rag.BM25PolicyRetriever", "rag")
    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]:
        query_terms = set(tokenize(query))
        scored: list[RetrievalHit] = []
        document_count = len(self._documents)
        if document_count == 0:
            return []

        for chunk, document, frequencies in zip(
            self.chunks, self._documents, self._frequencies, strict=True
        ):
            if category and chunk.categories and category not in chunk.categories:
                continue
            score = 0.0
            matched: list[str] = []
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                matched.append(term)
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_ratio = len(document) / self._average_length if self._average_length else 1.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append(
                    RetrievalHit(chunk=chunk, score=score, matched_terms=tuple(sorted(matched)))
                )

        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]


class HashingTextEncoder:
    """Dependency-free vector baseline; replace with a semantic encoder in production."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for term in tokenize(text):
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class FastEmbedTextEncoder:
    """Local semantic encoder backed by an ONNX BGE model through FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "本地语义 Embedding 需要安装 semantic 依赖：uv sync --extra semantic"
            ) from exc
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for raw in self.model.embed(list(texts)):
            vector = [float(value) for value in raw]
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class VectorPolicyRetriever:
    def __init__(self, chunks: list[PolicyChunk], encoder: TextEncoder) -> None:
        self.chunks = chunks
        self.encoder = encoder
        self._vectors = encoder.encode([chunk_text(chunk) for chunk in chunks])
        if len(self._vectors) != len(chunks):
            raise ValueError("encoder returned an unexpected number of vectors")

    @observed("rag.VectorPolicyRetriever", "rag")
    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]:
        query_vectors = self.encoder.encode([query])
        if not query_vectors:
            return []
        query_vector = query_vectors[0]
        scored: list[RetrievalHit] = []
        for chunk, vector in zip(self.chunks, self._vectors, strict=True):
            if category and chunk.categories and category not in chunk.categories:
                continue
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            if score > 0:
                scored.append(RetrievalHit(chunk=chunk, score=score, matched_terms=()))
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]


class ChromaPolicyRetriever:
    """Persistent Chroma vector index using the same provider-neutral encoder contract."""

    def __init__(
        self,
        chunks: list[PolicyChunk],
        encoder: TextEncoder,
        *,
        persist_directory: str | Path,
        collection_name: str | None = None,
    ) -> None:
        import chromadb

        self.chunks = {chunk.policy_id: chunk for chunk in chunks}
        self.encoder = encoder
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        fallback_identity = f"{type(encoder).__name__}:{getattr(encoder, 'dimensions', '')}"
        encoder_identity = str(getattr(encoder, "model_name", fallback_identity))
        identity_hash = hashlib.sha256(encoder_identity.encode("utf-8")).hexdigest()[:10]
        collection_name = collection_name or f"servicepilot_policies_{identity_hash}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if chunks:
            texts = [chunk_text(chunk) for chunk in chunks]
            embeddings = encoder.encode(texts)
            if len(embeddings) != len(chunks):
                raise ValueError("encoder returned an unexpected number of vectors")
            self.collection.upsert(
                ids=[chunk.policy_id for chunk in chunks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "title": chunk.title,
                        "categories": "|".join(chunk.categories),
                        "tags": "|".join(chunk.tags),
                    }
                    for chunk in chunks
                ],
            )

    @observed("rag.ChromaPolicyRetriever", "rag")
    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]:
        if not self.chunks or limit <= 0:
            return []
        query_vectors = self.encoder.encode([query])
        if not query_vectors:
            return []
        result = self.collection.query(
            query_embeddings=query_vectors,
            n_results=min(len(self.chunks), max(limit * 4, limit)),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[RetrievalHit] = []
        for policy_id, distance in zip(ids, distances, strict=True):
            chunk = self.chunks.get(policy_id)
            if chunk is None:
                continue
            if category and chunk.categories and category not in chunk.categories:
                continue
            score = max(0.0, 1.0 - float(distance))
            hits.append(RetrievalHit(chunk=chunk, score=score, matched_terms=()))
            if len(hits) == limit:
                break
        return hits


class QueryRewriter:
    REPLACEMENTS = {
        "钱没到": "退款到账时效",
        "怎么退": "退货退款申请条件",
        "坏掉": "商品故障质量问题",
        "摔碎": "商品破损质量问题",
        "快递不动": "物流停滞异常",
        "东西少了": "包裹少件漏发",
    }

    @observed("rag.query_rewrite", "rag")
    def rewrite(self, query: str) -> tuple[str, ...]:
        variants = [query]
        expanded = query
        for source, target in self.REPLACEMENTS.items():
            if source in expanded:
                expanded = expanded.replace(source, target)
        if expanded != query:
            variants.append(expanded)
        return tuple(variants)


@dataclass(frozen=True)
class RetrievalDiagnostics:
    cache_hit: bool
    rewritten_queries: tuple[str, ...]
    degraded_sources: tuple[str, ...]
    candidate_count: int
    reranker: str = "heuristic"
    reranker_degraded: bool = False


class AdvancedHybridPolicyRetriever:
    """Parallel multi-query recall with RRF, reranking, TTL cache, and degradation."""

    def __init__(
        self,
        lexical: PolicySearch,
        vector: PolicySearch,
        *,
        rewriter: QueryRewriter | None = None,
        candidate_limit: int = 12,
        rank_constant: int = 60,
        cache_ttl_seconds: float = 300.0,
        cache_capacity: int = 256,
        minimum_score: float = 0.01,
        reranker: PolicyReranker | None = None,
        clock=time.monotonic,
    ) -> None:
        self.lexical = lexical
        self.vector = vector
        self.rewriter = rewriter or QueryRewriter()
        self.candidate_limit = candidate_limit
        self.rank_constant = rank_constant
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_capacity = cache_capacity
        self.minimum_score = minimum_score
        self.reranker = reranker
        self.clock = clock
        self._cache: dict[tuple[str, str | None, int], tuple[float, list[RetrievalHit]]] = {}
        self.last_diagnostics = RetrievalDiagnostics(False, (), (), 0)
        self.reranker_failure_count = 0

    @observed("rag.AdvancedHybridPolicyRetriever", "rag")
    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]:
        key = (query, category, limit)
        now = self.clock()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            event("CACHE_HIT", source="rag", document_count=len(cached[1]))
            self.last_diagnostics = RetrievalDiagnostics(True, (query,), (), len(cached[1]))
            return list(cached[1])

        queries = self.rewriter.rewrite(query)
        jobs = [
            (source_name, variant, retriever)
            for variant in queries
            for source_name, retriever in (("bm25", self.lexical), ("vector", self.vector))
        ]
        rankings: list[list[RetrievalHit]] = []
        degraded: list[str] = []
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            futures = {
                submit_with_context(
                    pool, retriever.search,
                    variant,
                    category=category,
                    limit=self.candidate_limit,
                ): source_name
                for source_name, variant, retriever in jobs
            }
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    rankings.append(future.result())
                except Exception as exc:
                    event("RETRIEVAL_FAILED", source=source_name, error_type=type(exc).__name__)
                    degraded.append(source_name)

        with Span("rag.rrf", "rag", attributes={"ranking_count": len(rankings)}) as fusion:
            fused_scores: dict[str, float] = Counter()
            chunks: dict[str, PolicyChunk] = {}
            matched: dict[str, set[str]] = {}
            for ranking in rankings:
                for rank, hit in enumerate(ranking, start=1):
                    policy_id = hit.chunk.policy_id
                    fused_scores[policy_id] += 1.0 / (self.rank_constant + rank)
                    chunks[policy_id] = hit.chunk
                    matched.setdefault(policy_id, set()).update(hit.matched_terms)
    
            fusion.output({"scores": fused_scores, "candidate_count": len(fused_scores)})

        query_terms = set(tokenize(" ".join(queries)))
        reranked: list[RetrievalHit] = []
        for policy_id, rrf_score in fused_scores.items():
            chunk = chunks[policy_id]
            document_terms = set(tokenize(chunk_text(chunk)))
            overlap = len(query_terms & document_terms) / max(1, len(query_terms))
            score = rrf_score + 0.02 * overlap
            if score >= self.minimum_score:
                reranked.append(
                    RetrievalHit(
                        chunk=chunk,
                        score=score,
                        matched_terms=tuple(sorted(matched.get(policy_id, set()))),
                    )
                )
        with Span("rag.rerank", "rag", attributes={
            "source": type(self.reranker).__name__ if self.reranker else "heuristic"
        }) as rerank_span:
            ordered = sorted(reranked, key=lambda hit: hit.score, reverse=True)
            reranker_degraded = False
            if self.reranker and ordered:
                try:
                    result = self.reranker.rerank(query, ordered, limit=limit)
                except Exception:
                    reranker_degraded = True
                    self.reranker_failure_count += 1
                    result = ordered[:limit]
            else:
                result = ordered[:limit]
            rerank_span.output(result)
            rerank_span.set(degraded=reranker_degraded)
            if reranker_degraded:
                event("RETRIEVAL_FAILED", source="reranker")

        self._cache[key] = (now + self.cache_ttl_seconds, list(result))
        if len(self._cache) > self.cache_capacity:
            oldest = min(self._cache, key=lambda item: self._cache[item][0])
            self._cache.pop(oldest, None)
        self.last_diagnostics = RetrievalDiagnostics(
            False,
            queries,
            tuple(sorted(set(degraded))),
            len(fused_scores),
            type(self.reranker).__name__ if self.reranker else "heuristic",
            reranker_degraded,
        )
        return result


class HybridPolicyRetriever:
    """Fuse lexical and vector ranks using Reciprocal Rank Fusion."""

    def __init__(
        self,
        lexical: PolicySearch,
        vector: PolicySearch,
        *,
        rank_constant: int = 60,
        candidate_limit: int = 10,
    ) -> None:
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")
        self.lexical = lexical
        self.vector = vector
        self.rank_constant = rank_constant
        self.candidate_limit = candidate_limit

    @classmethod
    def from_json(
        cls, path: str | Path, *, encoder: TextEncoder | None = None
    ) -> HybridPolicyRetriever:
        chunks = load_policy_chunks(path)
        return cls(
            BM25PolicyRetriever(chunks),
            VectorPolicyRetriever(chunks, encoder or HashingTextEncoder()),
        )

    @observed("rag.HybridPolicyRetriever", "rag")
    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]:
        rankings = (
            self.lexical.search(query, category=category, limit=self.candidate_limit),
            self.vector.search(query, category=category, limit=self.candidate_limit),
        )
        scores: dict[str, float] = Counter()
        chunks: dict[str, PolicyChunk] = {}
        matched_terms: dict[str, set[str]] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                policy_id = hit.chunk.policy_id
                scores[policy_id] += 1.0 / (self.rank_constant + rank)
                chunks[policy_id] = hit.chunk
                matched_terms.setdefault(policy_id, set()).update(hit.matched_terms)

        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [
            RetrievalHit(
                chunk=chunks[policy_id],
                score=scores[policy_id],
                matched_terms=tuple(sorted(matched_terms[policy_id])),
            )
            for policy_id in ordered
        ]


# Backward-compatible name for the lexical baseline used by existing experiments.
PolicyRetriever = BM25PolicyRetriever
