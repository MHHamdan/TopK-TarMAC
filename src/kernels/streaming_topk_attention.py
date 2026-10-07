"""Streaming exact top-k attention: the score row is never resident.

Every fused score-then-select kernel in ``fused_topk_attention`` holds a query's
whole N-wide score row on chip before ranking it. That is not required by exact
top-k: selection only needs the k best candidates seen so far, so the keys can
be streamed in tiles exactly as a FlashAttention-style dense kernel streams
them, and the per-query state is a k-entry candidate buffer instead of an N-wide
row. This module is that kernel, for small k (the buffer is held in registers).

Design, per program (one tile of BLOCK_M queries of one batch element):

* The query tile is loaded once. Keys are walked in tiles of BLOCK_N; the score
  tile is formed with ``tl.dot`` at ``input_precision="ieee"`` (the same fp32
  IEEE path as the existing kernels) and scaled by 1/sqrt(d) after the product,
  as they do. The diagonal and out-of-range keys are excluded.
* Each score is packed with its key index into one int64,
  ``(ordered_int(score) << 32) | (0x7FFFFFFF - index)``, so a single signed
  integer order is "higher score first, then lower index first". The candidate
  buffer is BLOCK_M x KP of these words (KP = next power of two >= k; unused
  slots hold INT64_MAX so they are never evicted, empty slots distinct
  sentinels near INT64_MIN).
* Merge of a tile into the buffer, one of three schemes (constexpr, chosen per
  shape by scripts/autotune_streaming.py):
  - bitonic (default): ``tl.topk`` reduces the BLOCK_M x BLOCK_N tile to its
    KP best words, which are joined with the buffer and reduced to KP again.
    Data-independent, cost ~log^2(KP) + log(BLOCK_N/KP) compare-exchange
    stages per element.
  - rounds: k rounds of { per row, the tile's largest word replaces the
    buffer's smallest wherever it is larger; it leaves the tile }.
  - rounds with early exit: the same, but the loop stops as soon as no row in
    the program can still improve its set (at most k+1 checks per tile), so
    a tile below every row's current k-th score costs one max-reduction.
* After the last tile the buffer holds exactly the k largest words. Scores and
  indices are decoded from it, softmaxed over the k entries, and the k value
  rows are gathered and accumulated. Only the output is written.

Tie behaviour. The packed order makes the retained set the first k under
(score descending, index ascending): an exact tie at the k-th score keeps the
lower-indexed peer. ``torch.topk`` documents no tie rule; the existing
argmax-round kernel (``fused_topk_attention_tiled``) also keeps the lower index,
so the two fused kernels agree with each other exactly on ties, and with the
reference wherever the reference does the same. Exact ties at the k-th score
are counted in the verification output; with Gaussian inputs they are rare
(zero in every configuration benchmarked here).

Unlike the resident-row kernels this one has no cap on N.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


@triton.jit
def _ordered_key(x):
    """Monotone fp32 -> int32 map (a < b iff key(a) < key(b))."""
    i = x.to(tl.int32, bitcast=True)
    return i ^ ((i >> 31) & 0x7FFFFFFF)


@triton.jit
def _streaming_topk_kernel(
    Q, K, V, Out, Idx,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N, scale,
    TOPK: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_D: tl.constexpr,
    BITONIC: tl.constexpr = True,
    EARLY_EXIT: tl.constexpr = False,
    WRITE_IDX: tl.constexpr = False,
    PRECISION: tl.constexpr = "ieee",
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, KP)
    m_mask = offs_m < N
    d_mask = offs_d < HEAD_D

    q = tl.load(Q + b * stride_qb + offs_m[:, None] * stride_qn
                + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    I64_MIN: tl.constexpr = -9223372036854775808
    I64_MAX: tl.constexpr = 9223372036854775807
    # Empty slots: distinct sentinels below every real word, so the buffer's
    # minimum is always unique and exactly one slot is replaced per round.
    # Padding slots (k < KP): INT64_MAX, never the minimum, never evicted.
    init = tl.where(offs_k < TOPK, I64_MIN + 1 + offs_k.to(tl.int64), I64_MAX)
    buf = tl.broadcast_to(init[None, :], (BLOCK_M, KP))

    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        kblk = tl.load(K + b * stride_kb + offs_n[:, None] * stride_kn
                       + offs_d[None, :] * stride_kd,
                       mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(kblk), input_precision=PRECISION) * scale
        valid = n_mask[None, :] & (offs_n[None, :] != offs_m[:, None])
        # One int64 per candidate: score order first, then lower index first.
        word = (_ordered_key(s).to(tl.int64) << 32) | (
            0x7FFFFFFF - offs_n[None, :]).to(tl.int64)
        word = tl.where(valid, word, I64_MIN)
        if BITONIC:
            # Bitonic top-k of the tile, then of (buffer ++ tile top-k).
            if BLOCK_N > KP:
                word = tl.topk(word, KP, dim=1)
            both = tl.reshape(tl.join(buf, word), (BLOCK_M, 2 * KP))
            buf = tl.topk(both, KP, dim=1)
        else:
            # Max-and-replace rounds: the tile's best word replaces the
            # buffer's worst wherever it beats it, and leaves the tile.
            tmax = tl.max(word, axis=1)
            bmin = tl.min(buf, axis=1)
            need = tmax > bmin
            if EARLY_EXIT:
                # Stop once no row of the program can still improve its set
                # (at most TOPK replacements per tile, so this terminates).
                go = tl.max(need.to(tl.int32), axis=0)
                while go > 0:
                    buf = tl.where((buf == bmin[:, None]) & need[:, None],
                                   tmax[:, None], buf)
                    word = tl.where(word == tmax[:, None], I64_MIN, word)
                    tmax = tl.max(word, axis=1)
                    bmin = tl.min(buf, axis=1)
                    need = tmax > bmin
                    go = tl.max(need.to(tl.int32), axis=0)
            else:
                for _r in tl.static_range(TOPK):
                    buf = tl.where((buf == bmin[:, None]) & need[:, None],
                                   tmax[:, None], buf)
                    if _r + 1 < TOPK:
                        word = tl.where(word == tmax[:, None], I64_MIN, word)
                        tmax = tl.max(word, axis=1)
                        bmin = tl.min(buf, axis=1)
                        need = tmax > bmin

    keep = offs_k[None, :] < TOPK
    sel_idx = (0x7FFFFFFF - (buf & 0xFFFFFFFF)).to(tl.int32)
    hi = (buf >> 32).to(tl.int32)
    sel_s = (hi ^ ((hi >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    sel_s = tl.where(keep, sel_s, float("-inf"))
    row_max = tl.max(sel_s, axis=1)
    w = tl.where(keep, tl.exp(sel_s - row_max[:, None]), 0.0)
    denom = tl.sum(w, axis=1)

    if WRITE_IDX:
        # Verification only: the retained indices, (B, N, TOPK) contiguous.
        tl.store(Idx + (b * N + offs_m[:, None]) * TOPK + offs_k[None, :],
                 sel_idx, mask=m_mask[:, None] & keep)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for j in tl.static_range(TOPK):
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


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


# Best (BLOCK_M, BLOCK_N, num_warps, num_stages) per (d, N, k) found by
# scripts/autotune_streaming.py; read lazily from results/autotune_streaming.json.
_TUNED: dict | None = None


def _tuned_cfg(d: int, N: int, topk: int, batch: int):
    global _TUNED
    if _TUNED is None:
        import json
        from pathlib import Path
        f = Path(__file__).resolve().parents[2] / "results" / "autotune_streaming.json"
        _TUNED = {}
        if f.exists():
            for r in json.loads(f.read_text())["rows"]:
                if r.get("kind") == "streaming":
                    _TUNED[(r["head_dim"], r["n_agents"], r["k"],
                            r["batch"])] = r["best"]["cfg"]
    return _TUNED.get((d, N, topk, batch))


def streaming_topk_attention(q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor, topk: int,
                             block_m: int | None = None,
                             block_n: int | None = None,
                             num_warps: int | None = None,
                             num_stages: int | None = None,
                             tuned: bool = True,
                             bitonic: bool | None = None,
                             early_exit: bool | None = None,
                             precision: str = "ieee",
                             return_indices: bool = False):
    """Exact top-k attention with streamed keys. q, k, v: (B, N, d).

    With ``return_indices`` also returns the retained peer indices (B, N, k),
    for verification. Tiling defaults to the autotuned configuration for this (d, N, k, B) when
    one exists, else bitonic merge, BLOCK_M=64, BLOCK_N=16, 2 warps, 2 stages.
    """
    B, N, d = q.shape
    assert k.shape == v.shape == q.shape
    topk = max(1, min(topk, N - 1))
    assert topk <= 8, "register candidate buffer is sized for small k"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    cfg = _tuned_cfg(d, N, topk, B) if tuned else None
    cfg = dict(cfg or {"BLOCK_M": 64, "BLOCK_N": 16, "num_warps": 2,
                       "num_stages": 2})
    cfg.setdefault("BITONIC", True)
    cfg.setdefault("EARLY_EXIT", False)
    for key, val in (("BLOCK_M", block_m), ("BLOCK_N", block_n),
                     ("num_warps", num_warps), ("num_stages", num_stages),
                     ("BITONIC", bitonic), ("EARLY_EXIT", early_exit)):
        if val is not None:
            cfg[key] = val
    bd = max(16, _next_pow2(d))
    grid = (triton.cdiv(N, cfg["BLOCK_M"]), B)
    idx = (torch.empty(B, N, topk, device=q.device, dtype=torch.int32)
           if return_indices else out)
    _streaming_topk_kernel[grid](
        q, k, v, out, idx,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, 1.0 / (d ** 0.5),
        TOPK=topk,
        KP=_next_pow2(topk) if cfg["BITONIC"] else max(2, _next_pow2(topk)),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_D=bd,
        HEAD_D=d, BITONIC=cfg["BITONIC"],
        EARLY_EXIT=cfg["EARLY_EXIT"], WRITE_IDX=return_indices,
        PRECISION=precision,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return (out, idx) if return_indices else out
