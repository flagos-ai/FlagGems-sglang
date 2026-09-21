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
    S: tl.constexpr,
):
    pid = tl.program_id(0)
    root = tl.load(retrive_index_ptr + pid * stride_idx_b)
    tl.store(accept_index_ptr + pid * stride_acc_b, root)
    last_slot = root
    cur_row = 0
    num_accept = 0
    still = 1
    for step in range(1, S):
        if still:
            draft_token = tl.load(
                candidates_ptr + pid * stride_cand_b + step * stride_cand_s
            )
            p = tl.load(
                target_ptr
                + pid * stride_tp_b
                + cur_row * stride_tp_s
                + draft_token * stride_tp_v
            )
            q = tl.load(
                draft_ptr
                + pid * stride_dp_b
                + cur_row * stride_dp_s
                + draft_token * stride_dp_v
            )
            coin = tl.load(
                uniform_ptr + pid * stride_uni_b + (step - 1) * stride_uni_s
            )
            if coin * q < p:
                num_accept += 1
                tl.store(predicts_ptr + last_slot, draft_token)
                cur_row = step
                curr_slot = tl.load(
                    retrive_index_ptr
                    + pid * stride_idx_b
                    + step * stride_idx_s
                )
                tl.store(
                    accept_index_ptr
                    + pid * stride_acc_b
                    + num_accept * stride_acc_s,
                    curr_slot,
                )
                last_slot = curr_slot
            else:
                still = 0
    tl.store(accept_num_ptr + pid, num_accept)
    tl.store(last_slot_ptr + pid, last_slot)
    tl.store(cur_row_ptr + pid, cur_row)
    tl.store(all_accepted_ptr + pid, still)


@triton.jit
def _values_kernel(
    Values,
    Target,
    Draft,
    Rows,
    Accepted,
    S: tl.constexpr,
    V: tl.constexpr,
    FMAX: tl.constexpr,
):
    b = tl.program_id(0)
    idx = tl.program_id(1) * 128 + tl.arange(0, 128)
    row = tl.load(Rows + b)
    accepted = tl.load(Accepted + b)
    p = tl.load(Target + (b * S + row) * V + idx, idx < V, 0)
    q = tl.load(
        Draft + (b * (S - 1) + row) * V + idx, (idx < V) & (accepted == 0), 0
    ).to(tl.float32)
    limit = tl.full((), FMAX, tl.float32)
    q = tl.where(q != q, 0.0, q)
    q = tl.where(q == float("inf"), limit, q)
    q = tl.where(q == -float("inf"), -limit, q)
    corrected = tl.maximum(
        (p.to(tl.float32) - q).to(p.dtype).to(tl.float32), 0.0
    )
    val = tl.where(accepted != 0, p.to(tl.float32), corrected)
    tl.store(Values + b * V + idx, val, idx < V)


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
    _accept_kernel[batch,](
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
        S=steps,
        num_warps=4,
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
    sums = torch.empty((batch,), dtype=target.dtype, device=target.device)
    sum_parts = triton.cdiv(vocab, 16384)
    partials = torch.empty(
        (batch, sum_parts, 128), dtype=torch.float32, device=target.device
    )
    chunk = (
        triton.cdiv(vocab, 24 * 32) * 32
        if vocab >= 24 * 320
        else triton.cdiv(vocab, 16) * 16
    )
    parts = triton.cdiv(vocab, chunk)
    cdf = torch.empty_like(values)
    totals = torch.empty(
        (batch, parts), dtype=torch.float32, device=target.device
    )
    tokens = torch.empty(
        (batch, parts), dtype=torch.int32, device=target.device
    )
    _values_kernel[batch, triton.cdiv(vocab, 128)](
        values,
        target,
        draft,
        cur_row,
        all_accepted,
        S=steps,
        V=vocab,
        FMAX=max_value,
        enable_fp_fusion=False,
    )
    _recovered_sum_partials[batch, sum_parts](
        values, partials, vocab, sum_parts, enable_fp_fusion=False, num_warps=4
    )
    _recovered_sum_finish[batch,](
        partials, sums, sum_parts, enable_fp_fusion=False, num_warps=4
    )
    _compensated_cdf_blocks[batch, parts](
        values,
        cdf,
        totals,
        batch,
        vocab,
        chunk,
        parts,
        enable_fp_fusion=False,
        num_warps=4,
    )
    if parts > 1:
        _matrix_cdf_carry[batch, parts, triton.cdiv(chunk, 128)](
            cdf,
            totals,
            batch,
            vocab,
            chunk,
            parts,
            enable_fp_fusion=False,
            num_warps=4,
        )
    _recovered_sample_parts[batch, parts](
        cdf,
        sums,
        coins_final,
        tokens,
        vocab,
        chunk,
        parts,
        triton.next_power_of_2(chunk),
        enable_fp_fusion=False,
        num_warps=4,
    )
    _recovered_sample_finish[batch,](
        tokens,
        predicts_i,
        last_slot,
        parts,
        triton.next_power_of_2(parts),
        enable_fp_fusion=False,
        num_warps=4,
    )
    if predicts_i is not predicts:
        predicts.copy_(predicts_i.to(predicts.dtype))
    if accept_i is not accept_index:
        accept_index.copy_(accept_i.to(accept_index.dtype))
    return (predicts, accept_index, accept_num)


@triton.jit
def _ordered_positions(Positions, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    wide: tl.constexpr = (COUNT + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
    position_type: tl.constexpr = tl.int64 if wide else tl.int32
    pos = tl.program_id(0).to(position_type) * BLOCK + tl.arange(0, BLOCK).to(
        position_type
    )
    tl.store(Positions + pos, pos, pos < COUNT)


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
    pos = tl.program_id(0).to(position_type)
    b, step = (pos // S, pos % S)
    count = tl.load(Counts + b)
    valid = step <= count
    address = b.to(address_type) * IB + step.to(address_type) * IS
    slot = tl.load(Index + address, valid, 0)
    tl.store(Accept + pos, tl.where(valid, slot, -1))
    if valid:
        shadowed = tl.full((), 0, tl.int32)
        lane = tl.arange(0, BLOCK).to(position_type)
        for tile in range((pos + 1) // BLOCK, (COUNT + BLOCK - 1) // BLOCK):
            if shadowed == 0:
                other = tile * BLOCK + lane
                other_b = tl.where(other < COUNT, other // S, 0)
                other_step = other % S
                other_count = tl.load(Counts + other_b, other < COUNT, -1)
                relevant = (
                    (other > pos)
                    & (other < COUNT)
                    & (other_step <= other_count)
                )
                other_address = (
                    other_b.to(address_type) * IB
                    + other_step.to(address_type) * IS
                )
                other_slot = tl.load(Index + other_address, relevant, 0)
                shadowed = tl.max(
                    (relevant & (other_slot == slot)).to(tl.int32), 0
                )
        if shadowed == 0:
            token = tl.load(Local + pos)
            tl.store(Predicts + slot, token)


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
    positions = torch.empty(
        (batch, steps), dtype=position_dtype, device=candidates.device
    )
    _ordered_positions[triton.cdiv(count, 128),](positions, count, 128)
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
    _ordered_replay[count,](
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
def _recovered_sum_partials(
    Values, Partials, V: tl.constexpr, PARTS: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    lane = tl.arange(0, 128)
    total = tl.full((128,), 0, tl.float32)
    for block in range(tl.cdiv(tl.minimum(16384, V - part * 16384), 128)):
        offset = part * 16384 + block * 128 + lane
        total += tl.load(Values + row * V + offset, offset < V, 0).to(
            tl.float32
        )
    tl.store(Partials + (row * PARTS + part) * 128 + lane, total)


@triton.jit
def _recovered_sum_finish(Partials, Sums, PARTS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 16)
    t0 = tl.full((16,), 0, tl.float32)
    t1 = tl.full((16,), 0, tl.float32)
    t2 = tl.full((16,), 0, tl.float32)
    t3 = tl.full((16,), 0, tl.float32)
    t4 = tl.full((16,), 0, tl.float32)
    t5 = tl.full((16,), 0, tl.float32)
    t6 = tl.full((16,), 0, tl.float32)
    t7 = tl.full((16,), 0, tl.float32)
    for part in range(PARTS):
        base = Partials + (row * PARTS + part) * 128 + lane
        t0 += tl.load(base)
        t1 += tl.load(base + 16)
        t2 += tl.load(base + 32)
        t3 += tl.load(base + 48)
        t4 += tl.load(base + 64)
        t5 += tl.load(base + 80)
        t6 += tl.load(base + 96)
        t7 += tl.load(base + 112)
    left = t0 + t2 + (t1 + t3)
    right = t4 + t6 + (t5 + t7)
    high_left = tl.full((), 0, tl.float32)
    low_left = tl.full((), 0, tl.float32)
    high_right = tl.full((), 0, tl.float32)
    low_right = tl.full((), 0, tl.float32)
    for index in range(16):
        a = tl.sum(tl.where(lane == index, left, 0), 0)
        added_left = high_left + a
        back_left = added_left - high_left
        low_left += high_left - (added_left - back_left) + (a - back_left)
        high_left = added_left
        b = tl.sum(tl.where(lane == index, right, 0), 0)
        added_right = high_right + b
        back_right = added_right - high_right
        low_right += high_right - (added_right - back_right) + (b - back_right)
        high_right = added_right
    tl.store(Sums + row, high_left + low_left + (high_right + low_right))


@triton.jit
def _recovered_sample_parts(
    Cdf,
    Sums,
    Coins,
    Tokens,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    column = part * CHUNK + lane
    value = tl.load(
        Cdf + row * V + column, (lane < CHUNK) & (column < V), 0
    ).to(tl.float32)
    threshold = tl.load(Sums + row).to(tl.float32) * tl.load(Coins + row).to(
        tl.float32
    )
    match = (lane < CHUNK) & (column < V) & (value > threshold)
    token = tl.min(tl.where(match, column, V - 1), 0)
    tl.store(Tokens + row * PARTS + part, token)


@triton.jit
def _recovered_sample_finish(
    Tokens, Predicts, Slots, PARTS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    token = tl.load(Tokens + row * PARTS + part, part < PARTS, 2147483647)
    best = tl.min(token, 0)
    slot = tl.load(Slots + row)
    tl.store(Predicts + slot, best)


@triton.jit
def _compensated_cdf_blocks(
    Values,
    Cdf,
    Totals,
    B: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    lanes = tl.arange(0, 16)
    prefix = tl.full((16,), 0, tl.float32)
    suffix = tl.full((16,), 0, tl.float32)
    for block in range(tl.cdiv(CHUNK, 16)):
        high_p = prefix
        high_s = suffix
        low_p = tl.full((16,), 0, tl.float32)
        low_s = tl.full((16,), 0, tl.float32)
        for k in range(16):
            column = part * CHUNK + block * 16 + k
            x = tl.load(Values + row * V + column, (row < B) & (column < V), 0)
            x_p = tl.where(k <= lanes, x, 0.0)
            x_s = tl.where(k > lanes, x, 0.0)
            sum_p = high_p + x_p
            sum_s = high_s + x_s
            back_p = sum_p - high_p
            back_s = sum_s - high_s
            error_p = high_p - (sum_p - back_p) + (x_p - back_p)
            error_s = high_s - (sum_s - back_s) + (x_s - back_s)
            low_p += error_p
            low_s += error_s
            high_p = sum_p
            high_s = sum_s
        prefix = high_p + low_p
        columns = part * CHUNK + block * 16 + lanes
        tl.store(
            Cdf + row * V + columns, prefix + suffix, (row < B) & (columns < V)
        )
        suffix = high_s + low_s
    total = tl.sum(tl.where(lanes == 15, prefix, 0), 0)
    tl.store(Totals + row * PARTS + part, total, row < B)


@triton.jit
def _matrix_cdf_carry(
    Cdf,
    Totals,
    B: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    offset = tl.program_id(2) * 128 + tl.arange(0, 128)
    carry = tl.full((), 0, tl.float32)
    for previous in range(part):
        carry += tl.load(Totals + row * PARTS + previous)
    columns = part * CHUNK + offset
    mask = (row < B) & (offset < CHUNK) & (columns < V)
    values = tl.load(Cdf + row * V + columns, mask, 0)
    tl.store(Cdf + row * V + columns, values + carry, mask)


__all__ = ["chain_speculative_sampling"]
