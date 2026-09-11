# Target Token Scoring — Testing Guide

Two ports exist on two branches:

| Branch | Runner | File | Test machine |
|---|---|---|---|
| `feat/target-token-scoring` | V1 (`gpu_model_runner.py`) | production default; what vllm-ascend patches | **910B4 NPU** (CANN 8.5.1, torch_npu 2.9.0, vllm-ascend 0.19.rc1) |
| `feat/target-token-scoring-mrv2` | mrv2 (`gpu/model_runner.py`) | experimental, gated by `use_v2_model_runner=true` | **A100 GPU** (CUDA env) |

> **Version-drift caveat (read first).** Both branches are written against
> `vllm-project/vllm` `main` (base commit `9521c60`). The NPU box runs
> `vllm-ascend 0.19.rc1`, which pins a specific vLLM tag — the `gpu_model_runner.py`
> and `InputBatch` APIs on that tag may differ from `main`. Before any e2e NPU run,
> `git cherry-pick` the V1 commit (`670483d`) onto the vllm tree that
> `vllm-ascend 0.19.rc1` actually ships, and reconcile any drift (import names,
> `ExecuteModelState` field, `apply_grammar_bitmask` signature). The **unit tests
> and the demo below do not depend on the runner and run on CPU with no NPU/CANN** —
> run those first on any machine; they are the cheap correctness gate.

## 0. CPU gate (both machines, no GPU/CANN needed)

```bash
cd vllm
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e . --torch-backend=auto   # or the vllm-ascend wheel on NPU
uv pip install -r requirements/test/cuda.in

# Pure math + admission gate. No GPU. Must all pass.
.venv/bin/python -m pytest tests/v1/worker/test_target_token_scoring.py -v

# Synthetic equivalence (torch + numpy). Prints "equivalence OK".
.venv/bin/python examples/target_token_scoring/qwen2_5_reranker_demo.py

# Linters (88-char, ruff, mypy)
pre-commit run --files vllm/v1/worker/target_token_scoring/ \
  vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu/model_runner.py \
  vllm/config/model.py vllm/engine/arg_utils.py vllm/sampling_params.py \
  tests/v1/worker/test_target_token_scoring.py
```

If the unit tests fail here, **stop** — the math is wrong and no hardware run
matters. Paste the failure into the PR; do not proceed to e2e.

## 1. V1 e2e on 910B4 NPU

```bash
# On the NPU box, with vllm-ascend 0.19.rc1 + CANN 8.5.1 + torch_npu 2.9.0.
# Resolve the version drift (see caveat), then:
.venv/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B', target_token_scoring=True)
tok = llm.get_tokenizer()
yes_id, no_id = tok.convert_tokens_to_ids(['yes', 'no'])
sp = SamplingParams(temperature=0, max_tokens=1,
                    logprob_token_ids=[yes_id, no_id],
                    target_token_scoring_normalization='target_set')
out = llm.generate(['Is relevance scoring useful?'], sp)
print(out[0].outputs[0].logprobs)
"
```

**Equivalence check (compact vs native):** run the same prompt twice — once with
`target_token_scoring=True` (compact path) and once with
`target_token_scoring=False` (native full-vocab) — and compare the candidate
logprobs. They will **not** be identical (different normalization: `target_set`
denominator is K candidates, native `full_vocab` denominator is V), but the
**argmax winner** and the **rank order** of the K candidates must match. If you
need a bit-identical reference, set `target_token_scoring_normalization='full_vocab'`
on the compact run too — admission will then fall back to native (no compact),
confirming the flag is inert unless `target_set` is requested.

**Latency:** single-card scoring, K=2..8, batch 16-64. Expect the full-vocab
MatMul (`[B,H]@[V,H]^T`, V≈151k for Qwen2.5) to drop out of the hot path.
Measure with `vllm bench` or `time.perf_counter()` around `llm.generate`.

## 2. V2 (mrv2) e2e on A100 GPU

```bash
# mrv2 is GPU-experimental. Enable the V2 runner explicitly.
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B', target_token_scoring=True)
tok = llm.get_tokenizer()
yes_id, no_id = tok.convert_tokens_to_ids(['yes', 'no'])
sp = SamplingParams(temperature=0, max_tokens=1,
                    logprob_token_ids=[yes_id, no_id],
                    target_token_scoring_normalization='target_set')
out = llm.generate(['Is relevance scoring useful?'], sp)
print(out[0].outputs[0].logprobs)
"
```

If `VLLM_USE_V2_MODEL_RUNNER` is not the actual env var, check
`vllm/v1/worker/gpu_worker.py` for the `use_v2_model_runner` toggle and set it
via config.

### V2 runtime-correctness risk zones (verify, don't assume)

The compact path **bypasses `self.sampler`** on mrv2. These downstream consumers
were written assuming the native sampler ran; flag any anomaly here:

1. **`postprocess_sampled` → `post_update`** reads
   `self.sampler.penalties_state.output_bin_counts` and
   `self.req_states.last_sampled_tokens`. The compact path doesn't touch
   `penalties_state` (admission guarantees greedy, no penalties/frequency — so
   it should be inert), but verify no `IndexError` or stale-count drift across a
   multi-step run.
2. **`num_sampled = ones`**: the compact path emits `num_sampled == 1` for every
   request, assuming a pure scoring wave (every request done prefilling). If a
   mid-prefill request shares the wave, the native path emits `num_sampled == 0`
   for it via `get_num_sampled_and_rejected`; the compact path does not. Watch
   for this on mixed-batch e2e — ideally test with a wave where every request is
   at its scoring position.
3. **`AsyncOutput` / `ModelRunnerOutput`** assembly reads
   `sampler_output.sampled_token_ids`, `num_sampled`, `num_rejected` — all
   provided by `compact_sample_mrv2`; verify the async D2H copy doesn't choke on
   the compact `LogprobsTensors` (same type as V1, so low risk).
4. **`prompt_logprobs_worker.compute_prompt_logprobs(...)`** runs separately on
   `self.model.compute_logits` — for a scoring request it's empty, but confirm
   it doesn't re-invoke a full-vocab projection that defeats the speedup.

If any of 1-4 misbehaves on A100, **report which zone** and fall back: the V1
branch on NPU is the production-safe path; V2 is exploratory.

## 3. What to paste into the PR

- pytest summary from step 0 (all pass).
- demo output (`equivalence OK` for torch + numpy).
- V1 NPU: argmax/rank-order match vs native; latency before/after.
- V2 A100: same, plus an explicit note on which risk zone (1-4) held up or broke.
