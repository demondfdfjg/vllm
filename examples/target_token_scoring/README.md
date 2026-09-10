# Target Token Scoring — Qwen2.5 Reranker Demo

This demo shows the **compact scorer** fast path for fixed-candidate-set
scoring models (rerankers / relevance scoring), using a Qwen2.5-0.5B-style
yes/no reranker as the example.

## The problem this solves

vLLM already has `SamplingParams.logprob_token_ids` ("return logprobs for a
specific set of token ids") and a `generative_scoring` entrypoint. Both still
run the **full-vocab LM Head** (`[B, H] @ [V, H]^T -> [B, V]` + TP all-gather)
and then gather the K candidate columns in the sampler. `logprob_token_ids` is
a *sampler/output* semantic, not a sparse-LM-Head contract: seeing K results in
the response does **not** mean the device computed only K columns.

For low-latency single-step scoring, the full-vocab MatMul dominates.

## The compact fast path

With the engine flag on, an eligible wave skips the full-vocab projection:
`index_select` the K candidate weight rows (`[K, H]`) → compact `F.linear`
(`[B, K]`), then argmax→remap→target-set log-softmax, producing a
`SamplerOutput` directly. Ineligible waves fall back to native.

## Run the synthetic demo

```bash
.venv/bin/python examples/target_token_scoring/qwen2_5_reranker_demo.py
```

It checks `compact_logits[:, j] == full_vocab_logits[:, target_ids[j]]` and
shows caller-side score aggregation. A NumPy-only fallback runs when torch is
absent.

## Use it with a real Qwen2.5-0.5B model

```bash
vllm serve Qwen/Qwen2.5-0.5B --target-token-scoring
```

```python
from vllm import LLM, SamplingParams
# yes/no ids from the Qwen2.5 tokenizer
yes_id, no_id = ...
llm = LLM(model="Qwen/Qwen2.5-0.5B", target_token_scoring=True)
sp = SamplingParams(
    temperature=0, max_tokens=1, logprob_token_ids=[yes_id, no_id],
    # Opt this request into target-set semantics + the compact fast path.
    # Leaving the default ("full_vocab") keeps the native full-vocab path
    # unchanged, so the flag never silently changes a request's logprobs.
    target_token_scoring_normalization="target_set",
)
outputs = llm.generate(prompts, sp)
```

## Eligibility (else native fallback)

All must hold for the **whole wave**: every request sets identical ordered
`logprob_token_ids` **and** opts in with `target_token_scoring_normalization
="target_set"`; `max_tokens == 1`, `temperature == 0`, `n == 1`; no
`prompt_logprobs` / `logit_bias` / `allowed_token_ids` / `bad_words` /
structured output / speculative decoding / logits processors; dense unquantized
single-rank LM Head. A request that leaves normalization at the default
`"full_vocab"` always takes the native path. See
`docs/source/optimization/target_token_scoring.md`.
