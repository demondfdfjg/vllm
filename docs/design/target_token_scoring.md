# Target Token Scoring

## Motivation

A class of models — rerankers, relevance scorers, single-step classifiers —
finish prefill and only need the last position's hidden state projected onto a
**small, fixed set of K candidate tokens**, not the full vocabulary.

vLLM already has the primitives that *look* like this:

- `SamplingParams.logprob_token_ids` — "return logprobs for a specific set of
  token ids, more efficient than `logprobs=-1`."
- the `generative_scoring` entrypoint, which builds
  `SamplingParams(max_tokens=1, logprob_token_ids=...)`.

Both still run the **full-vocab LM Head**:

```
[B, H] @ [V, H]^T -> [B, V]   (+ tensor-parallel all-gather)
  -> sampler gathers the K candidate columns
```

`logprob_token_ids` is a *sampler/output* semantic, not a sparse-LM-Head
contract. Seeing K results in the response does **not** mean the device
computed only K columns. For low-latency single-step scoring, the full-vocab
MatMul and all-gather dominate latency.

## The compact fast path

With the engine flag `--target-token-scoring` on, an eligible wave replaces the
full-vocab projection:

```
[B, H] --index_select K weight rows--> [K, H] --F.linear--> [B, K]
  -> argmax (remap column j -> target_token_ids[j])
  -> target-set log-softmax
  -> SamplerOutput (shaped like gather_specific_token_logprobs output)
```

Ineligible waves fall back to the native path, unchanged.

## Opt-in contract (no silent semantic change)

The compact path is taken **only** when both hold:

- the engine flag `target_token_scoring` is on (server enables the feature); and
- the request **explicitly** sets
  `target_token_scoring_normalization="target_set"` (the request opts into
  target-set semantics).

The field defaults to `"full_vocab"` — the existing, unchanged behavior. So a
request that merely sets `logprob_token_ids` (expecting full-vocab logprobs)
always takes the native path, even with the flag on server-wide. The flag
cannot silently change the logprobs semantics of an existing request; a caller
must affirmatively request target-set semantics to get the compact path.

## Admission contract (wave-level)

The compact `[B, K]` tensor's column `j` is `target_token_ids[j]`, **not** vocab
id `j`. Because that shape is shared by the whole wave, eligibility is decided
once per wave, before `compute_logits`; any single failure downgrades the
**entire** wave to native (per-request mixing would let the generic sampler
mistake a compact column index for a vocab id).

All must hold:

- engine flag `target_token_scoring` on;
- every request has non-empty `logprob_token_ids`, identical and same order;
- every request opts in with `target_token_scoring_normalization == "target_set"`;
- `max_tokens == 1`, `temperature == 0` (greedy), `n == 1`;
- no `prompt_logprobs`, `logit_bias`, `allowed_token_ids`, `bad_words`,
  structured output, speculative decoding, or logits processors;
- LM Head is dense 2D, single-rank (`tp_size == 1`), unquantized, candidates in
  range.

## Normalization semantics

```
target_set :  log p_i = z_i - log(sum(exp(z_j))), j over the K candidates
full_vocab :  log p_i = z_i - log(sum(exp(z_j))), j over the full vocab V
```

These are different probability semantics, not approximations. Compact
`[B, K]` logits can compute `target_set` directly but cannot produce a
full-vocab denominator. `full_vocab` therefore falls back to native — and is
the default, so a request that wants full-vocab logprobs simply does not opt
in.

The runtime carries only ordered candidate values; caller-side aggregation
(threshold, label mapping, ranking) stays outside the runtime.

## Fallback boundary

Dense single-rank LM Head only. Native fallback (no silent wrong semantics)
for: TP vocab sharding, quantized/packed LM Head, vocab padding, custom output
layers, cross-rank candidate distribution, non-standard tied-embedding,
`full_vocab` normalization, any wave with a non-argmax-invariant processor.

## Usage

```bash
vllm serve Qwen/Qwen2.5-0.5B --target-token-scoring
```

```python
from vllm import SamplingParams
SamplingParams(temperature=0, max_tokens=1,
               logprob_token_ids=[yes_id, no_id],
               target_token_scoring_normalization="target_set")
```

## Verification

The compact path is correct iff `compact_logits[:, j] ==
full_vocab_logits[:, target_token_ids[j]]` (only irrelevant columns dropped).
See `examples/target_token_scoring/` and
`tests/v1/worker/test_target_token_scoring.py`.
