# DeepSeek online evaluation

Run date: 2026-09-16  
Model: `deepseek-flash`  
Base URL: `https://api.deepseek.com`  
Embedding mode: local hashing baseline (DeepSeek does not provide the project with an Embeddings endpoint)

## Intent routing

Dataset: 42 fixed, manually authored examples across 10 business intents plus cancel and unknown.

- Accuracy: 1.0000
- Macro-F1: 1.0000
- Raw report: `evaluations/results/intent_deepseek.json`

The first online run scored 0.9286 Accuracy and 0.9415 Macro-F1. Analysis showed that
an LLM `unknown` decision still contributed its 0.5 route weight to the denominator,
suppressing correct keyword decisions below the confidence threshold. The corrected
voter treats `unknown` as abstention and rejects Embedding-only decisions. The final
metrics are from a fresh run after that fix.

## End-to-end dialogue evaluation

Dataset: 20 fixed customer-service trajectories.

- Deterministic rule/tool checks: 20/20
- DeepSeek judgements: 20/20 completed
- Correctness: 4.50/5
- Groundedness: 4.50/5
- Safety: 5.00/5
- Coherence: 4.90/5
- Overall mean: 4.72/5
- Minimum per-case mean: 4.00/5
- Raw report: `evaluations/results/business_deepseek_judged.json`

Judge feedback led to concrete workflow changes: invoice guidance now uses order state
and policy citations, exchange and price-protection replies include eligibility guidance,
order switching explicitly explains pending-action cancellation, safety incidents instruct
the user to stop using the product, and shipped-address changes explain carrier forwarding
risks. The evaluator now supplies policy citations and sanitized tool results to the Judge.

## Reliability

LLM-as-Judge writes one checkpoint per trajectory, fingerprints the exact trajectory,
reuses unchanged judgements, and retries transient connection failures. This prevents a
single timeout from discarding completed requests or forcing the entire paid run to restart.

These are development-set results, not a claim of production accuracy. A larger held-out,
independently reviewed test set is still required before making generalization claims.
