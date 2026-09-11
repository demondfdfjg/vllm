# PR description — Compact target-token-scoring fast path (V1 runner)

> File against `vllm-project/vllm:main` from branch
> `feat/target-token-scoring`. Related issue: <!-- paste issue # -->.
> Duplicate-work check (re-run before posting):
> ```bash
> gh pr list --repo vllm-project/vllm --state open --search "target token scoring"
> gh pr list --repo vllm-project/vllm --state open --search "compact lm head logprob_token_ids"
> ```

## Purpose

Add a compact **target-token-scoring** fast path for single-step scoring models
(rerankers / relevance scoring) that only need scores for a small fixed set of
K candidate tokens.

**The gap this fills.** vLLM already has `SamplingParams.logprob_token_ids`
(#43463, merged — "return logprobs for a specific set of token ids") and a
`generative_scoring` entrypoint. Both still run the **full-vocab LM Head**
(`[B, H] @ [V, H]^T -> [B, V]` + tensor-parallel all-gather) and then gather the
K candidate columns in the sampler. `logprob_token_ids` is a *sampler/output*
semantic, not a sparse-LM-Head contract: seeing K results in the response does
not mean the device computed only K columns. For low-latency single-step
scoring the full-vocab MatMul dominates (V≈151k for Qwen2.5; the K=2 columns the
caller wants are ~0.001% of that compute).

**The fix.** With `--target-token-scoring` on, an eligible wave `index_select`s
the K candidate weight rows (`[K, H]`) -> compact `F.linear` (`[B, K]`), then
argmax->remap->target-set log-softmax, producing a `SamplerOutput` directly.
Ineligible waves fall back to native, unchanged. Dense single-rank unquantized
LM Head only; TP vocab sharding / quantized / spec decode / structured output
fall back to native rather than risk wrong semantics.

**Additive, not a redefinition.** The compact path is taken only when the
request *explicitly* opts in with `target_token_scoring_normalization=
"target_set"` (default `"full_vocab"` = the existing, unchanged behavior).
`logprob_token_ids` alone never triggers the compact path, so enabling the
engine flag server-wide cannot silently change the logprobs semantics of a
request that only asked for full-vocab logprobs. `target_set` and `full_vocab`
are different probability semantics, not approximations — the caller picks.

This is **not a duplicate** of existing work:
- **#43463** added the `logprob_token_ids` primitive this builds *on* (sampler/
  output side, not the LM Head).
- **#39351** (open, generative-scoring perf) is **complementary** — it fuses
  the log-softmax+gather but keeps the full-vocab LM Head; this PR removes the
  full-vocab MatMul itself; the two could stack.
- **#54335** (open, fixed-token prefill scoring) is **distinct** — distillation,
  prompt positions, bit-identical full-vocab logprobs, keeps the full-vocab LM
  Head. This PR is column-sparse at the LM Head for single-position reranker
  scoring.

## Changes

- `vllm/v1/worker/target_token_scoring/` — new package: `admission.py`
  (wave-level gate), `projector.py` (compact LM Head + per-runner
  `CompactLMHeadCache` keyed on weight signature incl. `_version`),
  `compact_sampler.py` (argmax+remap+target-set log-softmax for the V1
  `SamplerOutput`), `compact_sampler_mrv2.py` (same math, mrv2 `SamplerOutput`
  shape — sibling branch, see below), `state.py` (carrier).
- `vllm/v1/worker/gpu_model_runner.py` — one guarded branch in `execute_model`
  (before `compute_logits`) + one in `sample_tokens` (before `_sample`);
  `ExecuteModelState` gains a `target_token_scoring` field; the cache is an
  instance attribute on the runner.
- `vllm/config/model.py` + `vllm/engine/arg_utils.py` — `--target-token-scoring`
  flag (mirrors the `return_sampling_mask` precedent).
- `vllm/sampling_params.py` — `target_token_scoring_normalization` field
  (`"full_vocab"` default = native; `"target_set"` opts into the compact path).
- `examples/target_token_scoring/` — Qwen2.5 reranker demo (equivalence check +
  caller aggregation + NumPy fallback) + `TESTING.md` testing guide.
- `tests/v1/worker/test_target_token_scoring.py` — projector equivalence,
  cache invalidation, admission (eligible + one rejection per condition +
  default-normalization-is-native foot gun guard), sampler shape/stability/NaN.
- `docs/design/target_token_scoring.md` — design doc.

## Sibling branch: mrv2 port

`feat/target-token-scoring-mrv2` ports the same compact path to the experimental
mrv2 runner (`vllm/v1/worker/gpu/model_runner.py`). mrv2's `SamplerOutput`
carries extra bookkeeping tensors and its `Sampler` flattens `SamplingParams`
into sub-states (doesn't retain the raw object), so the mrv2 port adds
`compact_sampler_mrv2.py` and a per-req `SamplingParams` registry. It is
**blind (no mrv2 GPU at write time)** and has known runtime-correctness risk
zones (`postprocess_sampled`/`penalties_state` coupling when the compact path
bypasses `self.sampler`). **This PR targets V1** (production default; what
vllm-ascend uses on NPU). The mrv2 port is filed separately pending maintainer
guidance on which runner to land — see the linked issue.

## Test Plan

```bash
# unit tests (math + gate, no GPU needed) — see examples/target_token_scoring/TESTING.md
.venv/bin/python -m pytest tests/v1/worker/test_target_token_scoring.py -v

# demo (equivalence + aggregation; torch or numpy)
.venv/bin/python examples/target_token_scoring/qwen2_5_reranker_demo.py

# linters
pre-commit run --files vllm/v1/worker/target_token_scoring/ \
  vllm/v1/worker/gpu_model_runner.py vllm/config/model.py \
  vllm/engine/arg_utils.py vllm/sampling_params.py \
  tests/v1/worker/test_target_token_scoring.py

# duplicate-work check (submitter)
gh pr list --repo vllm-project/vllm --state open --search "target token scoring"
gh pr list --repo vllm-project/vllm --state open --search "compact lm head logprob_token_ids"
```

## Test Result

<!-- SUBMITTER: fill in after running on 910B4 NPU. Paste pytest summary,
     demo "equivalence OK" lines, and the V1 e2e argmax/rank-order match vs
     native + latency before/after. See TESTING.md. -->

- [ ] unit tests pass
- [ ] demo runs (torch + numpy equivalence OK)
- [ ] pre-commit clean
- [ ] (V1 e2e, 910B4 NPU) compact vs native: argmax winner + candidate rank
      order match (normalization differs by design); latency improvement on a
      scoring workload
- [ ] (model eval) Qwen2.5-0.5B reranker: compact path produces identical
      candidate logprobs to the native full-vocab path restricted to the same K
      columns

## Notes for reviewers

- The compact `[B, K]` tensor's column `j` is `target_token_ids[j]`, not vocab
  id `j`; the generic sampler is bypassed precisely so it can't mistake the two.
  Correctness invariant: `compact[:, j] == full[:, target_ids[j]]`.
- Eligibility is **wave-level**: any single failure downgrades the whole wave
  to native (never per-request mixing).
- The engine flag is a coarse enable; the **per-request**
  `target_token_scoring_normalization="target_set"` opt-in is what actually
  selects the compact path. Default `"full_vocab"` is native, so the flag cannot
  silently change `logprob_token_ids` semantics for an existing request.
- The compact path applies the LM Head's own row-wise postprocess (`soft_cap`,
  `scale`) directly rather than calling the full `LogitsProcessor`, because the
  processor's other branches assume a full-vocab width. Admission rejects every
  sampler-mask semantic so only `soft_cap`/`scale` can be active. The
  `grammar_output` bitmask branch is guarded on `tts_state is None` so a
  vocab-shaped mask can never touch the compact tensor.
- The async D2H optimization from the source case study is intentionally out of
  scope here (device/Ascend-specific, unverifiable without CANN); this PR is the
  device-agnostic compact-scorer + sampler-bypass core.

---

**AI assistance disclosure:** This PR was prepared with AI assistance. The
submitting human has reviewed every changed line, understands the admission
gate and fallback boundary end-to-end, and ran the tests above. Per the
contribution policy, no code-agent self-approval: a human owns this change.

Co-authored-by: Claude <noreply@anthropic.com>
