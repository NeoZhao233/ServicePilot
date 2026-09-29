# Online model mode

ServicePilot can use a Responses-compatible model for intent classification and LLM reranking, plus either an Embeddings-compatible API or a local BGE model for vector retrieval. Offline mode remains the default so tests and demos are deterministic and do not create API charges.

## Required protocol

The configured LLM service must provide this endpoint under `SERVICEPILOT_BASE_URL`:

- `POST /responses`, including strict JSON Schema output through `text.format`.
- `POST /embeddings`, accepting an array of input strings and returning indexed float vectors, is only required when `SERVICEPILOT_EMBEDDING_MODE=api`.

A provider that only implements Chat Completions is not currently compatible. A separate adapter should be added rather than adding provider-specific branches to `ServiceAgent`.

## Configuration

Copy the names from `.env.example` into the shell environment and replace the placeholder values. The application never reads a key from a committed configuration file.

```bash
export SERVICEPILOT_API_KEY='your-key'
export SERVICEPILOT_BASE_URL='https://api.openai.com/v1'
export SERVICEPILOT_INTENT_MODEL='your-model-id'
export SERVICEPILOT_EMBEDDING_MODEL='your-embedding-model-id'
```

### DeepSeek

DeepSeek's Responses API can drive intent routing and LLM-as-Judge. DeepSeek does not
document an Embeddings endpoint, so use the local retrieval encoder unless a separate
embedding provider adapter is configured:

```bash
export SERVICEPILOT_API_KEY='your-deepseek-key'
export SERVICEPILOT_BASE_URL='https://api.deepseek.com'
export SERVICEPILOT_INTENT_MODEL='deepseek-flash'
export SERVICEPILOT_EMBEDDING_MODE='local'
export SERVICEPILOT_EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
```

With `local`, FastEmbed runs a real 512-dimensional BGE semantic model through ONNX.
Install it with `uv sync --extra semantic`; the first run downloads the model. Set
`SERVICEPILOT_EMBEDDING_MODE=hashing` only for the dependency-free engineering baseline.

Run the terminal demo:

```bash
PYTHONPATH=src python3 -m servicepilot.cli --online
```

Run the API in online mode:

```bash
set -a; source .env; set +a
uv run servicepilot-web --online --port 8765
```

Run reproducible online comparisons:

```bash
PYTHONPATH=src python3 -m servicepilot.intent_evaluation --online
PYTHONPATH=src python3 -m servicepilot.retrieval_evaluation --online
```

The intent report includes a `fallback_count`. The retrieval report compares BM25,
Hashing Vector, semantic vector, semantic RRF, and Rewrite + semantic recall + LLM rerank.
If the reranker fails, retrieval degrades to the fused order and records the degradation.

## Failure behavior

- Invalid structured output becomes `unknown` instead of entering a business workflow.
- Low-confidence model output falls back to the rule router.
- An intent endpoint exception is contained and falls back to the rule router.
- Embeddings are generated when the policy index is built. If that endpoint is unavailable, online startup currently fails explicitly; it does not silently pretend that offline vectors are semantic embeddings.
- Business permissions and refund decisions remain deterministic regardless of model output.

## Live experiment checklist

Before publishing any online result, record:

1. Base URL provider and exact model IDs.
2. Prompt and JSON Schema version.
3. Dataset commit.
4. Raw predictions without credentials or personal data.
5. Accuracy, per-class F1, latency and token usage.
6. Number of requests that used rule fallback.
