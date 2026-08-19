"""Attention over a pre-chosen peer set, fused.

The pre-score measurements first reported here composed the attention stage from
library operators: gather the retained keys, gather the retained values,
contract, softmax, contract again. That composition writes and reads a
$(B,N,k,d)$ tensor twice, and the manuscript's own argument about the
score-then-select path is that such a constant is an upper bound rather than a
property of the design. Reporting it as a property would apply one standard to
the mechanism the paper criticises and another to the mechanism it proposes.

This kernel removes the composition. One program per tile of queries walks the
retained peers of each query, reading one key row and one value row per peer
directly out of global memory, accumulating the softmax-weighted sum with a
running maximum, and writing only the result. No gathered tensor exists.

Whether that is enough to make the escape case pay is the measurement; the
kernel exists so that the answer is about the algorithm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_prescore_kernel(
    Q, K, V, Idx, Out,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ib, stride_in, stride_ik,
    stride_ob, stride_on, stride_od,
    N, scale,
    KK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N
    d_mask = offs_d < HEAD_D

    q = tl.load(Q + b * stride_qb + offs_m[:, None] * stride_qn
                + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    run_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)

    # One pass over the retained peers, with an online softmax so that no score
    # vector is ever materialised.
    for j in tl.static_range(KK):
        idx = tl.load(Idx + b * stride_ib + offs_m * stride_in + j * stride_ik,
                      mask=m_mask, other=0)
        k_j = tl.load(K + b * stride_kb + idx[:, None] * stride_kn
                      + offs_d[None, :] * stride_kd,
                      mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        v_j = tl.load(V + b * stride_vb + idx[:, None] * stride_vn
                      + offs_d[None, :] * stride_vd,
                      mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        s = tl.sum(q * k_j, axis=1) * scale

        m_new = tl.maximum(run_max, s)
        alpha = tl.where(run_max == float("-inf"), 0.0,
                         tl.exp(run_max - m_new))
        p = tl.exp(s - m_new)
        denom = denom * alpha + p
        acc = acc * alpha[:, None] + p[:, None] * v_j
        run_max = m_new

    tl.store(Out + b * stride_ob + offs_m[:, None] * stride_on
             + offs_d[None, :] * stride_od,
             acc / denom[:, None],
             mask=m_mask[:, None] & d_mask[None, :])


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def fused_prescore_attention(q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor, idx: torch.Tensor,
                             block_m: int | None = None) -> torch.Tensor:
    """Attention over the peers named by ``idx``, with nothing intermediate
    written. q, k, v: (B, N, d); idx: (B, N, kk) int64 or int32."""
    B, N, d = q.shape
    kk = idx.shape[-1]
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    idx = idx.contiguous().to(torch.int32)
    out = torch.empty_like(q)
    bd = max(16, _next_pow2(d))
    if block_m is None:
        block_m = 64 if bd <= 64 else (32 if bd <= 128 else 16)
    block_m = min(block_m, max(16, _next_pow2(N)))
    grid = (triton.cdiv(N, block_m), B)
    _fused_prescore_kernel[grid](
        q, k, v, idx, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5),
        KK=kk, BLOCK_M=block_m, BLOCK_D=bd, HEAD_D=d,
        num_warps=4,
    )
    return out
