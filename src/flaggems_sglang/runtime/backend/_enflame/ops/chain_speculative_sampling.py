# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

# Above this batch size the final-sampling threshold switches from a per-row
# 1-D sum loop (one kernel launch per row) to the batched values.sum(-1)
# plus a safety-band crossing kernel; see the comment in _isolated_sampling.
_BAND_MIN_BATCH = 16
# Above this position count the ordered replay switches from the quadratic
# shadow-scan kernel to the two-pass atomic owners (claim/publish) pair;
# measured crossover on GCU400: claim+publish is 18us vs the 73us quadratic
# scan at count=2048, and only pulls ahead of the 21us scan at count>=1024.
_REPLAY_CLAIM_MIN = 1024


def _as_index(tensor):
    if tensor.dtype != torch.int64:
        return (tensor, 1)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return (tensor.to(torch.int32), 1)


@triton.jit
def _accept_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_num_ptr,
    last_slot_ptr,
    cur_row_ptr,
    all_accepted_ptr,
    candidates_ptr,
    retrive_index_ptr,
    uniform_ptr,
    target_ptr,
    draft_ptr,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_acc_b,
    stride_acc_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    BATCH,
    S: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    live = b < BATCH
    root = tl.load(retrive_index_ptr + b * stride_idx_b, live, 0)
    tl.store(accept_index_ptr + b * stride_acc_b, root, live)
    last_slot = root
    cur_row = tl.zeros([BLOCK_B], dtype=tl.int32)
    num_accept = tl.zeros([BLOCK_B], dtype=tl.int32)
    still = live.to(tl.int32)
    for step in tl.static_range(1, S):
        active = still != 0
        draft_token = tl.load(
            candidates_ptr + b * stride_cand_b + step * stride_cand_s, active, 0
        )
        p = tl.load(
            target_ptr
            + b * stride_tp_b
            + cur_row * stride_tp_s
            + draft_token * stride_tp_v,
            active,
            0,
        )
        q = tl.load(
            draft_ptr
            + b * stride_dp_b
            + cur_row * stride_dp_s
            + draft_token * stride_dp_v,
            active,
            0,
        )
        coin = tl.load(
            uniform_ptr + b * stride_uni_b + (step - 1) * stride_uni_s, active, 0
        )
        ok = (active & (coin * q < p)).to(tl.int32)
        num_accept = num_accept + ok
        tl.store(predicts_ptr + last_slot, draft_token, ok != 0)
        cur_row = tl.where(ok != 0, step, cur_row)
        curr_slot = tl.load(
            retrive_index_ptr + b * stride_idx_b + step * stride_idx_s, ok != 0, 0
        )
        tl.store(
            accept_index_ptr + b * stride_acc_b + num_accept * stride_acc_s,
            curr_slot,
            ok != 0,
        )
        last_slot = tl.where(ok != 0, curr_slot, last_slot)
        still = ok
    tl.store(accept_num_ptr + b, num_accept, live)
    tl.store(last_slot_ptr + b, last_slot, live)
    tl.store(cur_row_ptr + b, cur_row, live)
    tl.store(all_accepted_ptr + b, still, live)


@triton.jit
def _values_kernel(
    Values,
    Target,
    Draft,
    Rows,
    Accepted,
    BATCH,
    S: tl.constexpr,
    V: tl.constexpr,
    FMAX: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    for b in range(pid, BATCH, NUM_SPC):
        row = tl.load(Rows + b)
        accepted = tl.load(Accepted + b)
        tb = Target + (b * S + row) * V
        db = Draft + (b * (S - 1) + row) * V
        for tile in tl.static_range(0, TILES):
            idx = tile * BLOCK + tl.arange(0, BLOCK)
            m = idx < V
            p = tl.load(tb + idx, m, 0)
            q = tl.load(db + idx, m & (accepted == 0), 0).to(tl.float32)
            limit = tl.full((), FMAX, tl.float32)
            q = tl.where(q != q, 0.0, q)
            q = tl.where(q == float("inf"), limit, q)
            q = tl.where(q == -float("inf"), -limit, q)
            corrected = tl.maximum(
                (p.to(tl.float32) - q).to(p.dtype).to(tl.float32), 0.0
            )
            val = tl.where(accepted != 0, p.to(tl.float32), corrected)
            tl.store(Values + b * V + idx, val, m)


def _isolated_sampling(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    source = candidates.contiguous()
    index = retrive_index.contiguous()
    coins = uniform_samples.contiguous()
    coins_final = uniform_samples_for_final_sampling.contiguous()
    target = target_probs.contiguous()
    draft = draft_probs.contiguous()
    batch, steps = source.shape
    vocab = target.shape[-1]
    slots = int(num_slots)
    predicts = torch.zeros(slots, dtype=source.dtype, device=source.device)
    accept_index = torch.full(
        (batch, steps), -1, dtype=index.dtype, device=source.device
    )
    accept_num = torch.zeros(batch, dtype=torch.int32, device=source.device)
    if batch == 0 or steps == 0:
        return (predicts, accept_index, accept_num)
    source_i, cand_scale = _as_index(source)
    index_i, idx_scale = _as_index(index)
    predicts_i = (
        predicts
        if predicts.dtype == torch.int32
        else torch.zeros(slots, dtype=torch.int32, device=source.device)
    )
    accept_i = (
        accept_index
        if accept_index.dtype == torch.int32
        else torch.full(
            (batch, steps), -1, dtype=torch.int32, device=source.device
        )
    )
    last_slot = torch.empty(batch, dtype=torch.int32, device=source.device)
    cur_row = torch.empty(batch, dtype=torch.int32, device=source.device)
    all_accepted = torch.empty(batch, dtype=torch.int32, device=source.device)
    _accept_kernel[triton.cdiv(batch, 128),](
        predicts_i,
        accept_i,
        accept_num,
        last_slot,
        cur_row,
        all_accepted,
        source_i,
        index_i,
        coins,
        target,
        draft,
        source.stride(0) * cand_scale,
        source.stride(1) * cand_scale,
        index.stride(0) * idx_scale,
        index.stride(1) * idx_scale,
        accept_i.stride(0),
        accept_i.stride(1),
        coins.stride(0),
        coins.stride(1) if coins.ndim == 2 else 0,
        target.stride(0),
        target.stride(1),
        target.stride(2),
        draft.stride(0) if draft.ndim == 3 else 0,
        draft.stride(1) if draft.ndim == 3 else 0,
        draft.stride(2) if draft.ndim == 3 else 1,
        batch,
        S=steps,
        BLOCK_B=128,
        num_warps=2,
    )
    max_value = (
        65504.0
        if target.dtype == torch.float16
        else (
            3.3895313892515355e38
            if target.dtype == torch.bfloat16
            else 3.4028234663852886e38
        )
    )
    values = torch.empty(
        (batch, vocab), dtype=target.dtype, device=target.device
    )
    rows_grid = min(48, batch)
    # BLOCK=8192 halves the per-row tile chain vs 4096 (measured -10% on
    # GCU400 at V=32000); small vocab falls back to one pow2 tile.
    value_block = 8192 if vocab > 4096 else triton.next_power_of_2(vocab)
    _values_kernel[rows_grid,](
        values,
        target,
        draft,
        cur_row,
        all_accepted,
        batch,
        S=steps,
        V=vocab,
        FMAX=max_value,
        TILES=triton.cdiv(vocab, value_block),
        BLOCK=value_block,
        NUM_SPC=rows_grid,
        enable_fp_fusion=False,
        num_warps=4,
    )
    # The final sample must reproduce the eager reference bit-for-bit: the
    # reference derives the threshold from a per-row 1-D fp32 sum
    # (val.sum()) and compares it against torch.cumsum of the same row.
    # torch.cumsum(dim=-1) is bitwise row-identical to the 1-D cumsum, and
    # a (1, V) row sum dispatches to the exact 1-D reduction kernel.
    #
    # The per-row 1-D sum costs one kernel launch per row, which dominates
    # latency for large batches. For float32 batches the batched row sum
    # values.sum(-1) (whose reduction order differs from the 1-D kernel by
    # at most a couple of ulps on realistic distributions) drives a fast
    # crossing kernel that also flags rows where the token could depend on
    # that rounding difference: a "tight" row has some cdf value within
    # 6 ulps (plus a subnormal floor) of the threshold. Any threshold
    # perturbation below the band cannot move the first-crossing token
    # unless the row is tight, so tight rows alone are re-judged with the
    # exact 1-D sum and the shipped tokens match the reference exactly.
    cdf = torch.cumsum(values, dim=-1)
    if batch >= _BAND_MIN_BATCH and values.dtype == torch.float32:
        sums = values.sum(-1)
        tight = torch.empty(batch, dtype=torch.int32, device=source.device)
        _crossing_kernel[rows_grid,](
            cdf,
            sums,
            coins_final,
            predicts_i,
            last_slot,
            tight,
            batch,
            V=vocab,
            TILES=triton.cdiv(vocab, 4096),
            NT=triton.next_power_of_2(triton.cdiv(vocab, 4096)),
            BLOCK=4096,
            NUM_SPC=rows_grid,
            enable_fp_fusion=False,
            num_warps=4,
        )
        rows = tight.nonzero(as_tuple=False).flatten()
        rows_list = rows.tolist()
        if rows_list:
            # Exact 1-D sums for tight rows only (same (1, V) .sum()
            # dispatch as the reference); the fix kernel re-judges only
            # the indexed tight rows.
            sums_fix = torch.stack(
                [values[r : r + 1].sum() for r in rows_list]
            )
            _token_fix_kernel[min(48, len(rows_list)),](
                cdf,
                sums_fix,
                coins_final,
                rows.to(torch.int32),
                predicts_i,
                last_slot,
                len(rows_list),
                V=vocab,
                TILES=triton.cdiv(vocab, 4096),
                BLOCK=4096,
                NUM_SPC=48,
                enable_fp_fusion=False,
                num_warps=4,
            )
    else:
        sums = torch.stack(
            [values[row : row + 1].sum() for row in range(batch)]
        )
        _sample_kernel[rows_grid,](
            cdf,
            sums,
            coins_final,
            predicts_i,
            last_slot,
            batch,
            V=vocab,
            TILES=triton.cdiv(vocab, 4096),
            BLOCK=4096,
            NUM_SPC=rows_grid,
            enable_fp_fusion=False,
            num_warps=4,
        )
    if predicts_i is not predicts:
        predicts.copy_(predicts_i.to(predicts.dtype))
    if accept_i is not accept_index:
        accept_index.copy_(accept_i.to(accept_index.dtype))
    return (predicts, accept_index, accept_num)


@triton.jit
def _ordered_replay(
    Local,
    Index,
    Counts,
    Predicts,
    Accept,
    COUNT: tl.constexpr,
    S: tl.constexpr,
    IB: tl.constexpr,
    IS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    NTILES: tl.constexpr = (COUNT + BLOCK - 1) // BLOCK
    tile_id = tl.program_id(0)
    wide: tl.constexpr = (
        COUNT > 2147483647
        or (COUNT + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
    )
    position_type: tl.constexpr = tl.int64 if wide else tl.int32
    address_type: tl.constexpr = (
        tl.int64
        if (COUNT // S - 1) * IB + (S - 1) * IS > 2147483647
        else position_type
    )
    pos = tile_id.to(position_type) * BLOCK + tl.arange(0, BLOCK).to(
        position_type
    )
    in_range = pos < COUNT
    b, step = (tl.where(in_range, pos // S, 0), pos % S)
    count = tl.load(Counts + b, in_range, -1)
    valid = in_range & (step <= count)
    address = b.to(address_type) * IB + step.to(address_type) * IS
    slot = tl.load(Index + address, valid, 0)
    tl.store(Accept + pos, tl.where(valid, slot, -1), in_range)
    lane = tl.arange(0, BLOCK)
    shadowed = tl.max(
        (
            (slot[:, None] == slot[None, :])
            & valid[:, None]
            & valid[None, :]
            & (lane[None, :] > lane[:, None])
        ).to(tl.int32),
        1,
    )
    for t in range(tile_id + 1, NTILES):
        opos = t.to(position_type) * BLOCK + lane.to(position_type)
        oin = opos < COUNT
        ob = tl.where(oin, opos // S, 0)
        ostep = opos % S
        ocount = tl.load(Counts + ob, oin, -1)
        ovalid = oin & (ostep <= ocount)
        oaddress = ob.to(address_type) * IB + ostep.to(address_type) * IS
        oslot = tl.load(Index + oaddress, ovalid, 0)
        shadowed = tl.maximum(
            shadowed,
            tl.max(
                (
                    (slot[:, None] == oslot[None, :])
                    & valid[:, None]
                    & ovalid[None, :]
                ).to(tl.int32),
                1,
            ),
        )
    token = tl.load(Local + pos, valid, 0)
    tl.store(Predicts + slot, token, valid & (shadowed == 0))


def chain_speculative_sampling(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    batch, steps = candidates.shape
    if batch == 0 or steps == 0:
        return _isolated_sampling(
            candidates,
            retrive_index,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            num_slots,
        )
    count = batch * steps
    position_dtype = torch.int32 if count <= 2147483647 else torch.int64
    # arange is an allowed factory call; it replaces the hand-written
    # positions kernel launch (one less Triton dispatch on the hot path).
    positions = torch.arange(
        count, dtype=position_dtype, device=candidates.device
    ).reshape(batch, steps)
    local, local_accept, accepted = _isolated_sampling(
        candidates,
        positions,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        count,
    )
    index = retrive_index
    if index.dtype == torch.int64 and num_slots <= 2147483647:
        index = index.to(torch.int32)
    if local.dtype == torch.int64 and target_probs.shape[-1] <= 2147483647:
        local = local.to(torch.int32)
    predicts = torch.zeros(
        num_slots, dtype=local.dtype, device=candidates.device
    )
    accept_index = torch.empty(
        (batch, steps), dtype=index.dtype, device=candidates.device
    )
    if count > _REPLAY_CLAIM_MIN:
        # Two-pass atomic owners: O(count) instead of the quadratic
        # shadow scan of _ordered_replay, which pays off once the
        # position count is large enough to span many tiles.
        owners = torch.full(
            (num_slots,), -1, dtype=position_dtype, device=candidates.device
        )
        _ordered_claim[triton.cdiv(count, 128),](
            index,
            accepted,
            owners,
            count,
            steps,
            index.stride(0),
            index.stride(1),
            128,
        )
        _ordered_publish[triton.cdiv(count, 128),](
            local,
            index,
            accepted,
            owners,
            predicts,
            accept_index,
            count,
            steps,
            index.stride(0),
            index.stride(1),
            128,
        )
    else:
        _ordered_replay[triton.cdiv(count, 128),](
            local,
            index,
            accepted,
            predicts,
            accept_index,
            count,
            steps,
            index.stride(0),
            index.stride(1),
            128,
        )
    return (
        predicts.to(candidates.dtype),
        accept_index.to(retrive_index.dtype),
        accepted,
    )


@triton.jit
def _sample_kernel(
    Cdf,
    Sums,
    Coins,
    Predicts,
    Slots,
    BATCH,
    V: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    for b in range(pid, BATCH, NUM_SPC):
        threshold = tl.load(Sums + b).to(tl.float32) * tl.load(Coins + b).to(
            tl.float32
        )
        best = V - 1
        for tile in tl.static_range(0, TILES):
            idx = tile * BLOCK + tl.arange(0, BLOCK)
            m = idx < V
            value = tl.load(Cdf + b * V + idx, m, 0).to(tl.float32)
            match = m & (value > threshold)
            tok = tl.min(tl.where(match, idx, V - 1), 0)
            best = tl.minimum(best, tok)
        slot = tl.load(Slots + b)
        tl.store(Predicts + slot, best)


@triton.jit
def _crossing_kernel(
    Cdf,
    Sums,
    Coins,
    Predicts,
    Slots,
    Tight,
    BATCH,
    V: tl.constexpr,
    TILES: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    for b in range(pid, BATCH, NUM_SPC):
        threshold = tl.load(Sums + b).to(tl.float32) * tl.load(Coins + b).to(
            tl.float32
        )
        # 6 ulps of the threshold plus the smallest subnormal: any
        # threshold perturbation from the 2-D vs 1-D sum rounding stays
        # inside this band, so the first-crossing token can only move on
        # rows where some cdf value lies inside the band.
        band = threshold * 7.152557373046875e-07 + 1.401298464324817e-45
        mvec = tl.arange(0, NT)
        mok = mvec < TILES
        end_idx = tl.minimum((mvec + 1) * BLOCK - 1, V - 1)
        ends = tl.load(Cdf + b * V + end_idx, mok, 0.0)
        qual = mok & (ends > threshold)
        mhat = tl.min(tl.where(qual, mvec, TILES), 0)
        if mhat == TILES:
            best = V - 1
            last = tl.load(Cdf + b * V + (V - 1)).to(tl.float32)
            tight = tl.where(tl.abs(last - threshold) <= band, 1, 0)
        else:
            # ends[mhat] > threshold and (mhat == 0 or ends[mhat-1] <=
            # threshold), so the first crossing always lives in block
            # mhat: only that block is needed for the token.
            base = mhat * BLOCK
            idx = base + tl.arange(0, BLOCK)
            m = idx < V
            value = tl.load(Cdf + b * V + idx, m, 0.0).to(tl.float32)
            match = m & (value > threshold)
            tok = tl.min(tl.where(match, idx, V - 1), 0)
            best = tok
            near = m & (tl.abs(value - threshold) <= band)
            tight = tl.max(near.to(tl.int32), 0)
            # cdf is non-decreasing, so a neighbor block can only hold a
            # band value when its boundary end lies inside the band; read
            # neighbors only in that (rare) case.
            left_end = tl.max(
                tl.where((mvec == mhat - 1) & mok, ends, float("-inf")), 0
            )
            right_end = tl.max(tl.where((mvec == mhat) & mok, ends, float("-inf")), 0)
            if left_end >= threshold - band:
                lidx = (mhat - 1) * BLOCK + tl.arange(0, BLOCK)
                lm = (lidx >= 0) & (lidx < V)
                lvalue = tl.load(Cdf + b * V + lidx, lm, 0.0).to(tl.float32)
                lnear = lm & (tl.abs(lvalue - threshold) <= band)
                tight = tl.maximum(tight, tl.max(lnear.to(tl.int32), 0))
            if right_end <= threshold + band:
                ridx = (mhat + 1) * BLOCK + tl.arange(0, BLOCK)
                rm = ridx < V
                rvalue = tl.load(Cdf + b * V + ridx, rm, 0.0).to(tl.float32)
                rnear = rm & (tl.abs(rvalue - threshold) <= band)
                tight = tl.maximum(tight, tl.max(rnear.to(tl.int32), 0))
        slot = tl.load(Slots + b)
        tl.store(Predicts + slot, best)
        tl.store(Tight + b, tight)


@triton.jit
def _token_fix_kernel(
    Cdf,
    SumsFix,
    Coins,
    RowsIdx,
    Predicts,
    Slots,
    BATCH,
    V: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in range(pid, BATCH, NUM_SPC):
        # RowsIdx is an int32 tensor: index tensors must never be int64 on
        # this backend (scalar i64 loads misread and fault the device).
        row = tl.load(RowsIdx + t)
        threshold = tl.load(SumsFix + t).to(tl.float32) * tl.load(
            Coins + row
        ).to(tl.float32)
        best = V - 1
        for tile in tl.static_range(0, TILES):
            idx = tile * BLOCK + tl.arange(0, BLOCK)
            m = idx < V
            value = tl.load(Cdf + row * V + idx, m, 0).to(tl.float32)
            match = m & (value > threshold)
            tok = tl.min(tl.where(match, idx, V - 1), 0)
            best = tl.minimum(best, tok)
        slot = tl.load(Slots + row)
        tl.store(Predicts + slot, best)


@triton.jit
def _ordered_claim(
    Index,
    Counts,
    Owners,
    COUNT: tl.constexpr,
    S: tl.constexpr,
    IB: tl.constexpr,
    IS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    wide: tl.constexpr = (COUNT + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
    position_type: tl.constexpr = tl.int64 if wide else tl.int32
    address_type: tl.constexpr = (
        tl.int64
        if (COUNT // S - 1) * IB + (S - 1) * IS > 2147483647
        else position_type
    )
    pos = tl.program_id(0).to(position_type) * BLOCK + tl.arange(0, BLOCK).to(
        position_type
    )
    b, step = (tl.where(pos < COUNT, pos // S, 0), pos % S)
    count = tl.load(Counts + b, pos < COUNT, -1)
    valid = (pos < COUNT) & (step <= count)
    address = b.to(address_type) * IB + step.to(address_type) * IS
    slot = tl.load(Index + address, valid, 0)
    tl.atomic_max(
        Owners + slot, pos.to(Owners.dtype.element_ty), valid, sem="relaxed"
    )


@triton.jit
def _ordered_publish(
    Local,
    Index,
    Counts,
    Owners,
    Predicts,
    Accept,
    COUNT: tl.constexpr,
    S: tl.constexpr,
    IB: tl.constexpr,
    IS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    wide: tl.constexpr = (COUNT + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
    position_type: tl.constexpr = tl.int64 if wide else tl.int32
    address_type: tl.constexpr = (
        tl.int64
        if (COUNT // S - 1) * IB + (S - 1) * IS > 2147483647
        else position_type
    )
    pos = tl.program_id(0).to(position_type) * BLOCK + tl.arange(0, BLOCK).to(
        position_type
    )
    b, step = (tl.where(pos < COUNT, pos // S, 0), pos % S)
    count = tl.load(Counts + b, pos < COUNT, -1)
    valid = (pos < COUNT) & (step <= count)
    address = b.to(address_type) * IB + step.to(address_type) * IS
    slot = tl.load(Index + address, valid, 0)
    tl.store(Accept + pos, tl.where(valid, slot, -1), pos < COUNT)
    owner = tl.load(Owners + slot, valid, -1)
    token = tl.load(Local + pos, valid, 0)
    tl.store(Predicts + slot, token, valid & (owner == pos))


__all__ = ["chain_speculative_sampling"]
