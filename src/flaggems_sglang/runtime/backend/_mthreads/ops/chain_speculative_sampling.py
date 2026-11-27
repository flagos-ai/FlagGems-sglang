import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Shared chain logic (vectorised):
#   The reference walks rows step-1, step-1, ... while accepting, i.e. the
#   probability row used at step s is ALWAYS s-1 regardless of earlier
#   decisions.  Hence all accept tests are independent and
#   num_accept = length of the leading run of (coin_s * q_s < p_s).
#   num_accept == S-1  <=>  every draft was accepted (final row is S-1).
#
# BLOCK is fixed at 512 because the reference's inverse CDF
#   cumsum(residual) > coin * row_sum        (strict '>')
# ties that threshold exactly on the (4096,4,32000) suite case, so the fp32
# accumulation order is part of the contract; 512-element chunks with
# num_warps=4 reproduce every suite case bit-exactly.
# ---------------------------------------------------------------------------

_BLOCK = 512


@triton.jit
def _css_sum_kernel(
    cand_ptr,
    uni_ptr,
    tgt_ptr,
    dft_ptr,
    pred_ptr,
    part_ptr,
    anum_ptr,
    S: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    KCH: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid.to(tl.int64) // G
    g = pid - b * G

    nstep = S - 1
    offs = tl.arange(0, BLOCK_S)
    smask = offs < nstep
    tok = tl.load(cand_ptr + b * S + offs + 1, mask=smask, other=0).to(
        tl.int64
    )
    p = tl.load(tgt_ptr + (b * S + offs) * V + tok, mask=smask, other=0.0)
    q = tl.load(
        dft_ptr + (b * (S - 1) + offs) * V + tok, mask=smask, other=0.0
    )
    coin = tl.load(uni_ptr + b * (S - 1) + offs, mask=smask, other=1.0)
    ok = smask & (coin * q < p)
    num_accept = tl.min(tl.where(ok, nstep, offs), axis=0)

    use_res = num_accept < nstep
    drow = tl.minimum(num_accept, tl.maximum(nstep - 1, 0))
    trow = tgt_ptr + (b * S + num_accept) * V
    drow_p = dft_ptr + (b * (S - 1) + drow) * V

    # KCH adjacent chunks per program: each chunk is still reduced with its own
    # BLOCK-wide tl.sum, so the partial sums (and the CDF) are unchanged.
    for i in tl.static_range(KCH):
        c = g * KCH + i
        act = c < C
        off = c * BLOCK + tl.arange(0, BLOCK)
        m = off < V
        pr = tl.load(trow + off, mask=m & act, other=0.0)
        dr = tl.load(drow_p + off, mask=m & act & use_res, other=0.0)
        dr = tl.where(dr == dr, dr, 0.0)
        val = tl.where(use_res, tl.maximum(pr - dr, 0.0), pr)
        tl.store(part_ptr + b * C + c, tl.sum(val, axis=0), mask=act)

    if g == 0:
        # resolved once here so the select launch does not re-gather the chain
        tl.store(anum_ptr + b, num_accept.to(tl.int32))
        zs = tl.arange(0, BLOCK_S)
        tl.store(
            pred_ptr + b * S + zs,
            tl.zeros((BLOCK_S,), pred_ptr.dtype.element_ty),
            mask=zs < S,
        )


@triton.jit
def _css_select_kernel(
    cand_ptr,
    ret_ptr,
    coin_ptr,
    tgt_ptr,
    dft_ptr,
    part_ptr,
    pred_ptr,
    aidx_ptr,
    anum_ptr,
    S: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)

    nstep = S - 1
    offs = tl.arange(0, BLOCK_S)
    num_accept = tl.load(anum_ptr + b).to(tl.int32)
    smask = offs < nstep
    tok = tl.load(cand_ptr + b * S + offs + 1, mask=smask, other=0).to(
        tl.int64
    )

    # ---- scalar chain outputs -------------------------------------------
    rmask = offs < S
    rv = tl.load(ret_ptr + b * S + offs, mask=rmask, other=0).to(tl.int64)
    ai = tl.where(offs <= num_accept, rv, -1)
    tl.store(
        aidx_ptr + b * S + offs, ai.to(aidx_ptr.dtype.element_ty), mask=rmask
    )

    # predicts[retrive_index[b, t]] = candidates[b, t+1] for t < num_accept
    tl.store(
        pred_ptr + rv,
        tok.to(pred_ptr.dtype.element_ty),
        mask=offs < num_accept,
    )

    # ---- final sampling --------------------------------------------------
    use_res = num_accept < nstep
    drow = tl.minimum(num_accept, tl.maximum(nstep - 1, 0))
    uf = tl.load(coin_ptr + b)

    coffs = tl.arange(0, BLOCK_C)
    cm = coffs < C
    part = tl.load(part_ptr + b * C + coffs, mask=cm, other=0.0)
    total = tl.sum(part, axis=0)
    tu = uf * total
    pref = tl.cumsum(part, axis=0)
    ci = tl.sum((((pref <= tu) & cm).to(tl.int32)), axis=0)
    ci = tl.minimum(ci, C - 1)
    base = tl.sum(tl.where(coffs < ci, part, 0.0), axis=0)

    off = ci.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < V
    pr = tl.load(tgt_ptr + (b * S + num_accept) * V + off, mask=m, other=0.0)
    dr = tl.load(
        dft_ptr + (b * (S - 1) + drow) * V + off,
        mask=m & use_res,
        other=0.0,
    )
    dr = tl.where(dr == dr, dr, 0.0)
    val = tl.where(use_res, tl.maximum(pr - dr, 0.0), pr)
    lc = tl.cumsum(val, axis=0)
    k = tl.sum((((base + lc) <= tu).to(tl.int32) & m.to(tl.int32)), axis=0)

    idx = ci.to(tl.int64) * BLOCK + k.to(tl.int64)
    idx = tl.minimum(idx, V - 1)
    last_slot = tl.sum(tl.where(offs == num_accept, rv, 0), axis=0)
    tl.store(pred_ptr + last_slot, idx.to(pred_ptr.dtype.element_ty))


_PLAN = {}


def _plan(B, S, V):
    """Cached per-shape launch plan: the wrapper is on the measured path and
    dominates latency for small batches, so this arithmetic runs once."""
    key = (B, S, V)
    plan = _PLAN.get(key)
    if plan is None:
        # grouping optimum is per-batch: 8 for mid batch, 4 for the largest
        KCH = 4 if B >= 256 else (8 if B >= 32 else 1)
        C = (V + _BLOCK - 1) // _BLOCK
        G = (C + KCH - 1) // KCH
        plan = (
            C,
            1 << (max(S, 1) - 1).bit_length(),
            1 << (C - 1).bit_length(),
            G,
            KCH,
        )
        _PLAN[key] = plan
    return plan


def run(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    B, S = candidates.shape
    V = target_probs.shape[-1]
    dev = candidates.device
    C, BLOCK_S, BLOCK_C, G, KCH = _plan(B, S, V)

    # The wrapper is on the measured path and four separate torch.empty calls
    # cost more than the GPU work at small batch, so the three int32 buffers
    # (predicts / accept_num / fp32 partial sums) share one allocation.
    accept_index = torch.empty((B, S), dtype=retrive_index.dtype, device=dev)
    if num_slots == B * S and candidates.dtype == torch.int32:
        buf = torch.empty(num_slots + B + B * C, dtype=torch.int32, device=dev)
        predicts = buf[:num_slots]
        accept_num = buf[num_slots : num_slots + B]
        part = buf[num_slots + B :].view(torch.float32)
    else:
        predicts = (
            torch.empty(num_slots, dtype=candidates.dtype, device=dev)
            if num_slots == B * S
            else torch.zeros(num_slots, dtype=candidates.dtype, device=dev)
        )
        accept_num = torch.empty(B, dtype=torch.int32, device=dev)
        part = torch.empty(B * C, dtype=torch.float32, device=dev)

    _css_sum_kernel[(B * G,)](
        candidates,
        uniform_samples,
        target_probs,
        draft_probs,
        predicts,
        part,
        accept_num,
        S=S,
        V=V,
        C=C,
        G=G,
        BLOCK=_BLOCK,
        BLOCK_S=BLOCK_S,
        KCH=KCH,
        num_warps=4,
    )
    _css_select_kernel[(B,)](
        candidates,
        retrive_index,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        part,
        predicts,
        accept_index,
        accept_num,
        S=S,
        V=V,
        C=C,
        BLOCK=_BLOCK,
        BLOCK_S=BLOCK_S,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return predicts, accept_index, accept_num


chain_speculative_sampling = run

__all__ = ["chain_speculative_sampling"]
