# Issue draft — Compact (column-sparse) LM Head fast path for fixed-candidate-set scoring

> Paste the below into a new issue at `vllm-project/vllm`. Then immediately
> attach the PR. Replace the two fork-branch URLs with your live branches once
> pushed.

**Title:** [Feature] Compact LM Head fast path for fixed-candidate-set scoring (rerankers / relevance scoring)

**Labels:** `performance`, `module:sampling`, `module:worker`

---

## Motivation

A class of models — rerankers, relevance scorers, single-step classifiers —
finish prefill and only need the last position's hidden state projected onto a
**small, fixed set of K candidate tokens** (often K=2: yes/no), not the full
vocabulary.

vLLM already has the primitives that *look* like this:

- `SamplingParams.logprob_token_ids` (#43463, merged) — "return logprobs for a
  specific set of token ids, more efficient than `logprobs=-1`."
- the `generative_scoring` entrypoint, which builds
  `SamplingParams(max_tokens=1, logprob_token_ids=...)`.

Both still run the **full-vocab LM Head**:

```
[B, H] @ [V, H]^T -> [B, V]   (+ tensor-parallel all-gather on sharded heads)
  -> the sampler then gathers the K candidate columns
```

`logprob_token_ids` is a **sampler/output** semantic, not a sparse-LM-Head
contract: seeing K results in the response does **not** mean the device computed
only K columns. For low-latency single-step scoring on a single card, the
full-vocab MatMul dominates latency (V≈151k for Qwen2.5; the K=2 columns the
caller wants are ~0.001% of that compute).

## Proposal

A compact fast path that, on an eligible wave, replaces the full-vocab
projection:

```
[B, H] --index_select K candidate weight rows--> [K, H] --F.linear--> [B, K]
  -> argmax (remap column j -> target_token_ids[j])
  -> target-set log-softmax
  -> SamplerOutput (shaped like gather_specific_token_logprobs output)
```

Ineligible waves fall back to the native path, unchanged.

### Opt-in (no silent semantic change)

Two-level opt-in:

- engine flag `--target-token-scoring` (coarse enable); and
- per-request `target_token_scoring_normalization="target_set"` (actual opt-in).

The field defaults to `"full_vocab"` — the existing, unchanged behavior. So a
request that merely sets `logprob_token_ids` (expecting full-vocab logprobs)
always takes the native path, even with the flag on server-wide. The flag
cannot silently change the logprobs semantics of an existing request.

`target_set` and `full_vocab` are **different probability semantics, not
approximations**:

```
target_set :  log p_i = z_i - log(sum(exp(z_j))),  j over the K candidates
full_vocab :  log p_i = z_i - log(sum(exp(z_j))),  j over the full vocab V
```

Compact `[B, K]` logits can compute `target_set` directly but cannot produce a
full-vocab denominator; `full_vocab` therefore falls back to native — and is the
default.

### Wave-level admission gate

The compact `[B, K]` tensor's column `j` is `target_token_ids[j]`, **not** vocab
id `j`. Because that shape is shared by the whole wave, eligibility is decided
once per wave, before `compute_logits`; any single failure downgrades the
**entire** wave to native (per-request mixing would let the generic sampler
mistake a compact column index for a vocab id). All must hold:

- engine flag on; every request sets identical ordered `logprob_token_ids`
  **and** opts in with `target_token_scoring_normalization == "target_set"`;
- `max_tokens == 1`, `temperature == 0` (greedy), `n == 1`;
- no `prompt_logprobs` / `logit_bias` / `allowed_token_ids` / `bad_words` /
  structured output / speculative decoding / logits processors;
- dense, single-rank (`tp_size == 1`), unquantized LM Head, candidates in range.

## Why this is not a duplicate

- **#43463** (merged) added the `logprob_token_ids` primitive this builds *on* —
  it changed the sampler/output, not the LM Head. This PR is the LM-Head-side
  counterpart.
- **#39351** (open, "[Perf] generative scoring") is **complementary**: it
  optimizes the log-softmax + gather for full-vocab scoring via a fused Triton
  kernel but **keeps the full-vocab LM Head**. This PR removes the full-vocab
  MatMul itself; the two could stack.
- **#54335** (open, "[Feature] Add fixed-token prefill scoring") is
  **distinct**: it targets distillation (prompt positions, bit-identical
  full-vocab logprobs, reuses `compute_token_logprobs` gather) and keeps the
  full-vocab LM Head. It targets mrv2. This PR is column-sparse at the LM Head
  for single-position reranker scoring — different numerics, different goal.

## Question for maintainers (placement)

I have **two ports** ready on my fork and am unsure which runner to target:

- **V1** — `vllm/v1/worker/gpu_model_runner.py` (production default; what
  `vllm-ascend` patches/uses on NPU). Tested on Ascend 910B4.
  Branch: `https://github.com/demondfdfjg/vllm/tree/feat/target-token-scoring`
- **V2 (mrv2)** — `vllm/v1/worker/gpu/model_runner.py` (experimental, "under
  active development"). Ported but blind (no mrv2 GPU at write time); has known
  runtime-correctness risk zones around `postprocess_sampled`/`penalties_state`
  coupling when the compact path bypasses `self.sampler`.
  Branch: `https://github.com/demondfdfjg/vllm/tree/feat/target-token-scoring-mrv2`

Which runner do you want this on? I'll file the PR against the one you prefer and
drop/redesign the other. If V1 is preferred I can keep the mrv2 port as a follow-up.

## Footprint

- new package `vllm/v1/worker/target_token_scoring/` (admission gate, compact
  projector + per-runner cache, compact sampler for V1 + mrv2 output shapes,
  state carrier) — runner-agnostic;
- one guarded branch in each runner's `sample()` + `execute_model`;
- `--target-token-scoring` flag (mirrors `return_sampling_mask`);
- `target_token_scoring_normalization` SamplingParams field (default
  `full_vocab`);
- unit tests (projector equivalence, cache invalidation, admission incl.
  default-normalization-is-native guard, sampler shape/stability/NaN);
- Qwen2.5 reranker demo + design doc + testing guide.

The compact path is correct iff `compact_logits[:, j] ==
full_vocab_logits[:, target_token_ids[j]]` (only irrelevant columns dropped).
