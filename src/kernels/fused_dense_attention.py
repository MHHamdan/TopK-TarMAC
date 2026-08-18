"""A dense attention kernel written the way the sparse kernels are written.

Comparing a hand-written sparse kernel against a vendor library's dense kernel
confounds two differences: sparsity, and who wrote the code. At the head widths
used in multi-agent communication -- 16 to 64, rather than the 64 to 128 a
language model uses -- a vendor kernel tuned for long sequences and wide heads
need not be near its own best, so a sparse kernel can appear to win for reasons
that have nothing to do with sparsity.

This module removes the confound. It is a streaming attention kernel in the
manner of FlashAttention: one program per tile of queries, keys and values
walked in blocks, an online softmax carrying a running maximum and normaliser,
and no score matrix written to memory. It is written in the same framework, by
the same author, in the same session as the sparse kernels it is compared with.

It is deliberately *not* subject to the sparse kernels' structural constraint.
Exact score-then-select requires the complete score row of a query to be present
before the row can be ranked, which forces the sparse kernels to hold an
$N$-wide tile in registers and caps their occupancy. Dense attention has no such
requirement, and a baseline hobbled by a constraint it does not have would
understate what dense attention costs. That asymmetry -- streaming is available
to dense attention and not to exact top-k -- is itself one of the reasons the
sparse path does not win, and it is reported as a finding rather than hidden in
a baseline.

The dense figure reported in the manuscript is the better of this kernel and
PyTorch's fused scaled-dot-product path, so no sparse implementation is credited
with a win that is really a deficiency in the baseline.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_dense_kernel(
    Q, K, V, Out,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_KN: tl.constexpr,
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
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start in range(0, N, BLOCK_KN):
        offs_n = start + tl.arange(0, BLOCK_KN)
        n_mask = offs_n < N
        k_blk = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn
                        + offs_d[None, :] * stride_kd,
                        mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v_blk = tl.load(V + b * stride_vb + offs_n[:, None] * stride_vn
                        + offs_d[None, :] * stride_vd,
                        mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k_blk), input_precision="ieee") * scale
        valid = n_mask[None, :] & (offs_n[None, :] != offs_m[:, None])
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A query whose whole block is masked keeps m_i = -inf; guard the
        # exponent so the running state stays finite.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                                m_i - m_safe))
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        p = tl.exp(s - m_safe[:, None])
        p = tl.where(valid, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
        m_i = m_new

    tl.store(Out + b * stride_ob + offs_m[:, None] * stride_on
             + offs_d[None, :] * stride_od,
             acc / l_i[:, None],
             mask=m_mask[:, None] & d_mask[None, :])


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def fused_dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_m: int | None = None,
                          block_kn: int | None = None) -> torch.Tensor:
    """Streaming dense self-attention with the diagonal masked."""
    B, N, d = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    bd = max(16, _next_pow2(d))
    npn = max(16, _next_pow2(N))
    if block_m is None:
        block_m = 64 if bd <= 64 else (32 if bd <= 128 else 16)
    if block_kn is None:
        block_kn = 64 if bd <= 128 else 32
    block_m, block_kn = min(block_m, npn), min(block_kn, npn)
    # Shrink the tiles until the key and value blocks and the accumulator fit
    # the SM's shared-memory budget, rather than failing at the widest head.
    stages = 2
    while (2 * block_kn * bd + block_m * bd) * 4 * stages > 80_000:
        if stages > 1:
            stages = 1
        elif block_kn > 16:
            block_kn //= 2
        elif block_m > 16:
            block_m //= 2
        else:
            break
    grid = (triton.cdiv(N, block_m), B)
    _flash_dense_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5),
        BLOCK_M=block_m, BLOCK_KN=block_kn, BLOCK_D=bd, HEAD_D=d,
        num_warps=4, num_stages=stages,
    )
    return out
