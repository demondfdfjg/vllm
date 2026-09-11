"""mrv2 compact sampler: argmax over ``[B, K]`` -> mrv2 ``SamplerOutput``.

Same compact math as :mod:`compact_sampler` (argmax over the K candidate
columns, remap column ``j`` -> ``target_token_ids[j]``, target-set log-softmax),
but wraps the result in the mrv2 runner's ``SamplerOutput``
(:class:`vllm.v1.worker.gpu.sample.output.SamplerOutput`), which carries the
extra ``num_nans`` / ``num_sampled`` / ``num_rejected`` bookkeeping tensors that
``vllm/v1/worker/gpu/model_runner.py`` reads from the return of ``sample()``.

Normalization is fixed to ``target_set`` (log-softmax over the K candidates);
``full_vocab`` is rejected at admission.
"""

from __future__ import annotations

import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from .state import TargetTokenScoringState


def _target_set_logprobs(logits: torch.Tensor) -> torch.Tensor:
    """Stable log-softmax over the K candidate dimension (FP32)."""
    f = logits.float()
    m = f.amax(dim=-1, keepdim=True)
    # All-inf rows would produce NaN; clamp the max to 0 so the subtraction is
    # finite. Such rows are surfaced via num_nans rather than a wrong argmax.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = f - m
    return shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)


def compact_sample_mrv2(
    compact_logits: torch.Tensor,
    state: TargetTokenScoringState,
    *,
    compute_nans: bool,
) -> SamplerOutput:
    """Turn compact ``[B, K]`` logits into an mrv2 ``SamplerOutput``.

    Args:
        compact_logits: ``[B, K]`` projected logits (column j = candidate j).
        state: The wave's target-token-scoring contract (candidate ids).
        compute_nans: Whether the runner computes NaN counts (mirrors
            ``Sampler.compute_nans``); when ``False`` the native path sets
            ``num_nans=None`` and so does this path.

    Returns:
        An mrv2 ``SamplerOutput`` whose ``logprobs_tensors`` matches the shape
        the native ``compute_topk_scores`` would produce for the same
        ``logprob_token_ids``; ``num_sampled`` is ones (admission guarantees
        ``max_tokens == 1`` and a scoring step where every request is done
        prefilling), ``num_rejected`` is zeros, ``sampling_mask_tensors`` is
        ``None`` (admission rejects ``return_sampling_mask`` semantics).
    """
    num_reqs, k = compact_logits.shape
    target_ids = state.target_ids_tensor  # [K], on compact_logits.device
    device = compact_logits.device

    row_has_nan = torch.isnan(compact_logits).any(dim=-1)  # [B]

    selected_cols = compact_logits.argmax(dim=-1)  # [B]
    sampled = target_ids.gather(0, selected_cols)  # [B]
    # NaN rows cannot be trusted; keep the shape valid (column 0 fallback) and
    # let num_nans flag the request rather than fabricating an id.
    sampled = torch.where(
        row_has_nan, target_ids.gather(0, torch.zeros_like(selected_cols)), sampled
    )
    sampled_token_ids = sampled.unsqueeze(-1)  # [B, 1]

    logprobs = _target_set_logprobs(compact_logits)  # [B, K] (FP32)
    sampled_logprob = logprobs.gather(-1, selected_cols.unsqueeze(-1))  # [B,1]
    full_logprobs = torch.cat([sampled_logprob, logprobs], dim=-1)  # [B, K+1]

    token_ids_table = torch.empty(
        num_reqs, k + 1, dtype=torch.int32, device=device
    )
    token_ids_table[:, 0] = sampled
    token_ids_table[:, 1:] = target_ids.unsqueeze(0).to(torch.int32)

    ranks = torch.sum(
        (logprobs > sampled_logprob).to(torch.int64), dim=-1
    )  # [B]

    if row_has_nan.any():
        full_logprobs = full_logprobs.masked_fill(
            row_has_nan.unsqueeze(-1), float("nan")
        )

    logprobs_tensors = LogprobsTensors(
        logprob_token_ids=token_ids_table,
        logprobs=full_logprobs,
        selected_token_ranks=ranks,
    )

    # num_nans mirrors the native ``get_num_nans`` contract: a per-request int32
    # count. The compact tensor is [B, K] (not [B, V]) so a row with any NaN
    # counts as 1; clean rows count 0. Admission makes NaN implausible (clean
    # FP32 projection), so this is bookkeeping, not a hot path.
    num_nans = row_has_nan.to(torch.int32) if compute_nans else None

    # Admission guarantees max_tokens == 1 and a scoring step (every request
    # done prefilling), so each request samples exactly one token and rejects
    # none. num_rejected is zeros (no speculative decoding under admission).
    # RISK: if a request is mid-prefill in the same wave, the native path would
    # emit num_sampled == 0 for it; this compact path assumes a pure scoring
    # wave. The testing guide calls this out.
    num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)

    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        sampling_mask_tensors=None,
    )
