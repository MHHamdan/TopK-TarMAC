"""A fused select--gather--aggregate kernel for score-then-select top-k attention.

The manuscript's measured sparse path composes ``torch.topk``, ``Tensor.gather``
and a batched contraction out of general-purpose operators, and each of those
operators reads and writes the intermediate it is handed. A reviewer is entitled
to ask how much of the measured penalty is that composition rather than the
algorithm. This kernel answers the question by removing the composition: one
Triton program per (batch element, query) pair holds the whole score row in
registers, extracts the k largest entries there, and accumulates the softmax
weighted sum of the corresponding value rows without ever writing a score
matrix, a top-k index tensor or a gathered value tensor to global memory.

What it does *not* do is avoid forming the scores. Assumption A2 of the
manuscript's ceiling is that a score-then-select rule must evaluate every
candidate score before it can rank them, and this kernel evaluates every one of
them; it simply never stores them. That is the strongest form of the sparse
path available on this hardware, which is what makes it the right thing to
benchmark against dense.

Two implementations are provided:

``fused_topk_attention``
    Selection by ``k`` sequential argmax-and-mask rounds over the score row.
    Exact, and the cost of selection is explicit and proportional to ``k``.

``fused_topk_attention_tiled``
    The same, with a tile of queries per program so that the score product runs
    on the matrix-multiply units rather than as a per-query reduction. This is
    the strongest sparse implementation we were able to write.

``fused_topk_attention_masked``
    Selection by an exact bisection on the integer key of the score, so
    selection costs a fixed number of passes over the score row rather than one
    per retained peer. The retained set is expressed as a zero weight rather
    than by compaction, so the aggregation performs dense arithmetic whatever
    ``k`` is: this is the fused counterpart of the masked implementation, and it
    exists to show what the sparse kernels must give up to make selection cheap.
    Because it selects by threshold rather than by rank, an exact tie at the
    k-th largest score retains k+1 peers where ``torch.topk`` would break the tie
    by index. Measured incidence: one query row in 196,608 at N=512, k=64, d=64
    over twelve seeds. Every other row agrees with the reference to fp32
    tolerance.

Both require ``N`` to fit a single Triton block, which caps them at the 1024
agents this study covers. Nothing about the approach is specific to that cap --
a two-pass variant would lift it -- but the manuscript's regime does not need it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_topk_kernel(
    Q, K, V, Out,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N, scale,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    """One program per (batch element, query). Never writes an N-sized tensor."""
    pid = tl.program_id(0)
    b = pid // N
    i = pid % N

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_D
    n_mask = offs_n < N

    # q_i, held in registers for the whole program.
    q = tl.load(Q + b * stride_qb + i * stride_qn + offs_d * stride_qd,
                mask=d_mask, other=0.0)

    # The complete score row. This is assumption A2 made mechanical: every
    # candidate is scored before any is ranked. It is never written out.
    k_ptrs = K + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    kblk = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    scores = tl.sum(kblk * q[None, :], axis=1) * scale

    # Self-attention is masked, matching every other implementation in the study.
    neg = float("-inf")
    scores = tl.where(n_mask & (offs_n != i), scores, neg)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    denom = tl.zeros([1], dtype=tl.float32)
    running_max = tl.max(scores, axis=0)

    for _ in range(TOPK):
        j = tl.argmax(scores, axis=0)
        s = tl.max(scores, axis=0)
        w = tl.exp(s - running_max)
        v = tl.load(V + b * stride_vb + j * stride_vn + offs_d * stride_vd,
                    mask=d_mask, other=0.0)
        acc += w * v
        denom += w
        # Remove the taken entry so the next round finds the next largest.
        scores = tl.where(offs_n == j, neg, scores)

    out = acc / denom
    tl.store(Out + b * stride_ob + i * stride_on + offs_d * stride_od, out,
             mask=d_mask)


@triton.jit
def _float_to_ordered_int(x):
    """Monotone map from fp32 to int32: a < b iff key(a) < key(b).

    Bisecting on the score value itself cannot be exact -- a fixed number of
    halvings leaves a bracket that may be wider than the gap between the k-th
    and (k+1)-th score, in which case extra peers survive the threshold and the
    softmax support is wrong. Bisecting on the integer bit pattern terminates
    exactly, because the keys are integers and the search interval is finite.
    """
    i = x.to(tl.int32, bitcast=True)
    return i ^ ((i >> 31) & 0x7FFFFFFF)


@triton.jit
def _fused_topk_masked_kernel(
    Q, K, V, Out,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N, scale, topk,
    BISECT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    """Fused top-k whose selection cost does not grow with k.

    Selection is an exact bisection on the integer key of the score, so its cost
    is a fixed number of passes over the score row rather than k of them. The
    price is paid in the aggregation: the retained set is expressed as a weight
    of zero rather than by compaction, so this kernel performs the *dense*
    aggregation arithmetic regardless of k. It is the fused counterpart of the
    masked implementation, and the trade it embodies -- selection cheap in k, or
    arithmetic proportional to k, but not both -- is one of the paper's results
    rather than an incidental detail of it.
    """
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N
    n_mask = offs_n < N
    d_mask = offs_d < HEAD_D

    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for dc in tl.static_range(0, BLOCK_D, BLOCK_DC):
        offs_c = dc + tl.arange(0, BLOCK_DC)
        c_mask = offs_c < HEAD_D
        q_c = tl.load(Q + b * stride_qb + offs_m[:, None] * stride_qn
                      + offs_c[None, :] * stride_qd,
                      mask=m_mask[:, None] & c_mask[None, :], other=0.0)
        k_c = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn
                      + offs_c[None, :] * stride_kd,
                      mask=n_mask[:, None] & c_mask[None, :], other=0.0)
        scores += tl.dot(q_c, tl.trans(k_c), input_precision="ieee")
    scores = scores * scale

    neg = float("-inf")
    valid = n_mask[None, :] & (offs_n[None, :] != offs_m[:, None])
    scores = tl.where(valid, scores, neg)

    keys = _float_to_ordered_int(scores).to(tl.int64)
    row_max = tl.max(keys, axis=1)
    # Invariant: at least topk keys are >= lo, fewer than topk are > hi.
    lo = tl.min(tl.where(valid, keys, row_max[:, None]), axis=1)
    hi = row_max
    for _ in range(BISECT):
        mid = lo + (hi - lo + 1) // 2
        cnt = tl.sum(tl.where(keys >= mid[:, None], 1, 0), axis=1)
        lo = tl.where(cnt >= topk, mid, lo)
        hi = tl.where(cnt >= topk, hi, mid - 1)

    keep = keys >= lo[:, None]
    m = tl.max(tl.where(keep, scores, neg), axis=1)
    w = tl.where(keep, tl.exp(scores - m[:, None]), 0.0)
    denom = tl.sum(w, axis=1)

    # Aggregation over the whole row with zero weight on the discarded peers:
    # sparse in memory traffic, dense in arithmetic.
    for dc in tl.static_range(0, BLOCK_D, BLOCK_DC):
        offs_c = dc + tl.arange(0, BLOCK_DC)
        c_mask = offs_c < HEAD_D
        v_c = tl.load(V + b * stride_vb + offs_n[:, None] * stride_vn
                      + offs_c[None, :] * stride_vd,
                      mask=n_mask[:, None] & c_mask[None, :], other=0.0)
        part = tl.dot(w, v_c, input_precision="ieee")
        tl.store(Out + b * stride_ob + offs_m[:, None] * stride_on
                 + offs_c[None, :] * stride_od,
                 part / denom[:, None],
                 mask=m_mask[:, None] & c_mask[None, :])


@triton.jit
def _fused_topk_tiled_kernel(
    Q, K, V, Out,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N, scale,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    """Tile of queries per program, so the score product runs on tensor cores.

    The single-query kernel above computes each score row as a reduction, which
    leaves the matrix-multiply units idle. This variant loads BLOCK_M queries at
    once and forms their scores with tl.dot, so the score stage uses the same
    hardware path a dense GEMM would. Selection and aggregation then proceed per
    query row inside the tile, still without any intermediate reaching global
    memory. This is the strongest sparse implementation we were able to write,
    and it is the one the manuscript compares against dense attention.
    """
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N
    n_mask = offs_n < N
    d_mask = offs_d < HEAD_D

    # Every candidate is scored before any is ranked -- assumption A2 -- but on
    # the matrix-multiply path and without leaving the streaming multiprocessor.
    # The head dimension is walked in chunks so that the key tile resident in
    # shared memory stays inside the SM's budget at the largest (N, d) measured.
    # input_precision="ieee": TF32 would perturb the scores enough to change
    # which peers the top-k selects, so the kernel would no longer compute the
    # same function as the reference implementation it is timed against.
    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for dc in tl.static_range(0, BLOCK_D, BLOCK_DC):
        offs_c = dc + tl.arange(0, BLOCK_DC)
        c_mask = offs_c < HEAD_D
        q_c = tl.load(Q + b * stride_qb + offs_m[:, None] * stride_qn
                      + offs_c[None, :] * stride_qd,
                      mask=m_mask[:, None] & c_mask[None, :], other=0.0)
        k_c = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn
                      + offs_c[None, :] * stride_kd,
                      mask=n_mask[:, None] & c_mask[None, :], other=0.0)
        scores += tl.dot(q_c, tl.trans(k_c), input_precision="ieee")
    scores = scores * scale
    neg = float("-inf")
    valid = n_mask[None, :] & (offs_n[None, :] != offs_m[:, None])
    scores = tl.where(valid, scores, neg)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    running_max = tl.max(scores, axis=1)

    for _ in range(TOPK):
        j = tl.argmax(scores, axis=1)                     # (BLOCK_M,)
        s = tl.max(scores, axis=1)
        w = tl.exp(s - running_max)
        v = tl.load(V + b * stride_vb + j[:, None] * stride_vn
                    + offs_d[None, :] * stride_vd,
                    mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        acc += w[:, None] * v
        denom += w
        scores = tl.where(offs_n[None, :] == j[:, None], neg, scores)

    tl.store(Out + b * stride_ob + offs_m[:, None] * stride_on
             + offs_d[None, :] * stride_od,
             acc / denom[:, None],
             mask=m_mask[:, None] & d_mask[None, :])


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def fused_topk_attention_tiled(q: torch.Tensor, k: torch.Tensor,
                               v: torch.Tensor, topk: int,
                               block_m: int | None = None) -> torch.Tensor:
    """Tiled fused top-k attention; the score stage runs on tensor cores.

    ``block_m`` defaults to the largest query tile whose score tile, key chunk
    and accumulator fit the SM's shared-memory budget at this (N, d).
    """
    B, N, d = q.shape
    assert N <= 1024
    topk = max(1, min(topk, N - 1))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    bn, bd = _next_pow2(N), max(16, _next_pow2(d))
    # Keep the resident key chunk near 32 kB and the score tile near 32 kB, so
    # the whole working set fits the 100 kB shared-memory budget of an SM.
    bdc = max(16, min(bd, 32768 // (bn * 4)))
    if block_m is None:
        block_m = max(16, min(32, 32768 // (bn * 4)))
    block_m = min(block_m, bn)
    grid = (triton.cdiv(N, block_m), B)
    _fused_topk_tiled_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5),
        TOPK=topk, BLOCK_M=block_m, BLOCK_N=bn,
        BLOCK_D=bd, BLOCK_DC=bdc, HEAD_D=d,
        num_warps=4,
    )
    return out


def fused_topk_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         topk: int) -> torch.Tensor:
    """Fused score-then-select top-k attention. q, k, v: (B, N, d)."""
    B, N, d = q.shape
    assert k.shape == v.shape == q.shape
    assert N <= 1024, "single-block variant; N > 1024 needs a two-pass kernel"
    topk = max(1, min(topk, N - 1))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    _fused_topk_kernel[(B * N,)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5),
        TOPK=topk, BLOCK_N=_next_pow2(N), BLOCK_D=_next_pow2(d), HEAD_D=d,
        num_warps=4,
    )
    return out


def fused_topk_attention_masked(q: torch.Tensor, k: torch.Tensor,
                                v: torch.Tensor, topk: int,
                                bisect: int = 34) -> torch.Tensor:
    """Fused top-k with k-independent selection and dense aggregation.

    ``bisect`` must be large enough to close an interval of at most 2^32 integer
    keys; 34 halvings leave a bracket of width one, which is exact.
    """
    B, N, d = q.shape
    assert N <= 1024
    topk = max(1, min(topk, N - 1))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    bn, bd = _next_pow2(N), max(16, _next_pow2(d))
    bdc = max(16, min(bd, 32768 // (bn * 4)))
    block_m = max(16, min(32, 32768 // (bn * 4)))
    block_m = min(block_m, bn)
    grid = (triton.cdiv(N, block_m), B)
    _fused_topk_masked_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5), topk,
        BISECT=bisect, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_D=bd,
        BLOCK_DC=bdc, HEAD_D=d,
        num_warps=4,
    )
    return out


def reference_topk_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             topk: int) -> torch.Tensor:
    """The PyTorch gathered path, as the manuscript benchmarks it."""
    B, N, d = q.shape
    topk = max(1, min(topk, N - 1))
    scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    eye = torch.eye(N, device=q.device, dtype=torch.bool).unsqueeze(0)
    scores = scores.masked_fill(eye, float("-inf"))
    top_scores, top_idx = scores.topk(topk, dim=-1, sorted=False)
    w = torch.softmax(top_scores, dim=-1)
    idx = top_idx.reshape(B, N * topk, 1).expand(B, N * topk, d)
    v_g = v.gather(1, idx).view(B, N, topk, d)
    return torch.matmul(w.unsqueeze(-2), v_g).squeeze(-2)


@triton.jit
def _fused_topk_sort_kernel(
    Q, K, V, Out,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N, scale,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    """Selection by one sort, aggregation over the compacted survivors.

    The two selection strategies above trade against each other: argmax rounds
    keep the aggregation sparse and cost one row-wide reduction per retained
    peer, while threshold bisection costs a fixed number of passes and gives up
    compaction, so it performs the dense aggregation arithmetic. This kernel is
    the attempt to have both.

    Sorting the score row once, with each score packed against its own index in
    the low half of a 64-bit word, yields the retained peers *already
    compacted*: they are the first k entries of the sorted row, and their
    indices come out of the sort with them. Selection then costs one bitonic
    sort, whose depth is logarithmic in the agent count and independent of k,
    and the aggregation reads exactly k value rows. Nothing is written to memory
    but the result.

    The sort is over the whole row, so its register footprint is
    BLOCK_M x BLOCK_N 64-bit words; the launcher shrinks the query tile until
    that fits, which at the largest agent counts leaves a narrow tile.
    """
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N
    n_mask = offs_n < N
    d_mask = offs_d < HEAD_D

    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for dc in tl.static_range(0, BLOCK_D, BLOCK_DC):
        offs_c = dc + tl.arange(0, BLOCK_DC)
        c_mask = offs_c < HEAD_D
        q_c = tl.load(Q + b * stride_qb + offs_m[:, None] * stride_qn
                      + offs_c[None, :] * stride_qd,
                      mask=m_mask[:, None] & c_mask[None, :], other=0.0)
        k_c = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn
                      + offs_c[None, :] * stride_kd,
                      mask=n_mask[:, None] & c_mask[None, :], other=0.0)
        scores += tl.dot(q_c, tl.trans(k_c), input_precision="ieee")
    scores = scores * scale

    neg = float("-inf")
    valid = n_mask[None, :] & (offs_n[None, :] != offs_m[:, None])
    scores = tl.where(valid, scores, neg)

    # Pack the order-preserving integer key above the peer index, so that one
    # sort orders by score and carries the index along with it.
    keys = _float_to_ordered_int(scores).to(tl.int64)
    packed = (keys << 32) | tl.broadcast_to(
        offs_n[None, :].to(tl.int64) & 0xFFFFFFFF, (BLOCK_M, BLOCK_N))
    srt = tl.sort(packed, dim=1, descending=True)

    # The survivors are the leading k entries, already compacted.
    top = tl.gather(srt, tl.broadcast_to(offs_k[None, :], (BLOCK_M, BLOCK_K)), 1)
    sel_idx = (top & 0xFFFFFFFF).to(tl.int32)
    sel_key = (top >> 32).to(tl.int32)
    sel_scr = (sel_key ^ ((sel_key >> 31) & 0x7FFFFFFF)).to(tl.float32,
                                                            bitcast=True)
    keep = offs_k[None, :] < TOPK
    sel_scr = tl.where(keep, sel_scr, neg)

    row_max = tl.max(sel_scr, axis=1)
    w = tl.where(keep, tl.exp(sel_scr - row_max[:, None]), 0.0)
    denom = tl.sum(w, axis=1)

    # Aggregate over exactly the k retained rows. The reductions below are over
    # the compacted axis, which is k wide, not over the score row.
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for j in tl.static_range(BLOCK_K):
        take = offs_k[None, :] == j
        idx_j = tl.sum(tl.where(take, sel_idx, 0), axis=1)
        w_j = tl.sum(tl.where(take, w, 0.0), axis=1)
        v = tl.load(V + b * stride_vb + idx_j[:, None] * stride_vn
                    + offs_d[None, :] * stride_vd,
                    mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        acc += w_j[:, None] * v

    tl.store(Out + b * stride_ob + offs_m[:, None] * stride_on
             + offs_d[None, :] * stride_od,
             acc / denom[:, None],
             mask=m_mask[:, None] & d_mask[None, :])


def fused_topk_attention_sort(q: torch.Tensor, k: torch.Tensor,
                              v: torch.Tensor, topk: int,
                              block_m: int | None = None) -> torch.Tensor:
    """Fused top-k with sort-based selection: sparse aggregation, and selection
    whose cost does not grow with k."""
    B, N, d = q.shape
    assert N <= 1024
    topk = max(1, min(topk, N - 1))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    bn, bd = _next_pow2(N), max(16, _next_pow2(d))
    bk = max(16, _next_pow2(topk))
    bdc = max(16, min(bd, 32768 // (bn * 4)))
    if block_m is None:
        # The sort holds BLOCK_M x BLOCK_N 64-bit words; keep that near 32 kB.
        block_m = max(16, min(32, 32768 // (bn * 8)))
    block_m = min(block_m, bn)
    grid = (triton.cdiv(N, block_m), B)
    _fused_topk_sort_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5),
        TOPK=topk, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk,
        BLOCK_D=bd, BLOCK_DC=bdc, HEAD_D=d,
        num_warps=4,
    )
    return out
