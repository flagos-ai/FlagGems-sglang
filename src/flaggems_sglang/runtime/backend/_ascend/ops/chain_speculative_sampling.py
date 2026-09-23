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


def base_as_index(tensor):
    if tensor.dtype != torch.int64:
        return (tensor, 1)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return (tensor.view(torch.int32), 2)


@triton.jit
def base_accept_kernel(
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
def base_values_kernel(
    Values,
    Target,
    Draft,
    Rows,
    Accepted,
    S: tl.constexpr,
    V: tl.constexpr,
    FMAX: tl.constexpr,
    BLOCK: tl.constexpr,
    B: tl.constexpr,
):
    blocks: tl.constexpr = (V + BLOCK - 1) // BLOCK
    for work in range(tl.program_id(0), B * blocks, tl.num_programs(0)):
        b = work // blocks
        idx = work % blocks * BLOCK + tl.arange(0, BLOCK)
        row = tl.load(Rows + b)
        accepted = tl.load(Accepted + b)
        p = tl.load(Target + (b * S + row) * V + idx, idx < V, 0)
        val = p.to(tl.float32)
        if accepted == 0:
            q = tl.load(Draft + (b * (S - 1) + row) * V + idx, idx < V, 0).to(
                tl.float32
            )
            q = tl.where(q != q, 0.0, q)
            q = tl.minimum(tl.maximum(q, -FMAX), FMAX)
            val = tl.maximum((val - q).to(p.dtype).to(tl.float32), 0.0)
        tl.store(Values + b * V + idx, val, idx < V)


@triton.jit
def base_sum_kernel(
    Values,
    Sums,
    Predicts,
    Slots,
    V: tl.constexpr,
    NT: tl.constexpr,
    LOG_NT: tl.constexpr,
    WARP_LOG: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.arange(0, NT)
    a0 = tl.full([NT], 0.0, tl.float32)
    a1 = tl.full([NT], 0.0, tl.float32)
    a2 = tl.full([NT], 0.0, tl.float32)
    a3 = tl.full([NT], 0.0, tl.float32)
    for base in range(tl.cdiv(V, NT * 4)):
        if V > 128:
            idx = base * NT * 4 + t * 4
            delta: tl.constexpr = 1
            bound: tl.constexpr = V - V % 4
        else:
            idx = base * NT * 4 + t
            delta: tl.constexpr = NT
            bound: tl.constexpr = V
        a0 += tl.load(Values + b * V + idx, idx < bound, 0).to(tl.float32)
        a1 += tl.load(Values + b * V + idx + delta, idx + delta < bound, 0).to(
            tl.float32
        )
        a2 += tl.load(
            Values + b * V + idx + 2 * delta, idx + 2 * delta < bound, 0
        ).to(tl.float32)
        a3 += tl.load(
            Values + b * V + idx + 3 * delta, idx + 3 * delta < bound, 0
        ).to(tl.float32)
    if V > 128:
        idx = V - V % 4 + t
        a0 += tl.load(Values + b * V + idx, idx < V, 0).to(tl.float32)
    value = a0 + a1 + a2 + a3
    for j in tl.static_range(5, LOG_NT):
        offset = NT >> j - 4
        other = tl.gather(value, tl.minimum(t + offset, NT - 1), 0)
        value = tl.where(t < offset, value + other, value)
    for j in tl.static_range(0, WARP_LOG):
        other = tl.gather(value, tl.minimum(t + (1 << j), NT - 1), 0)
        value += other
    total = tl.sum(tl.where(t == 0, value, 0.0), 0)
    tl.store(Sums + b, total)
    slot = tl.load(Slots + b)
    tl.store(Predicts + slot, V - 1)


@triton.jit
def base_group_prefix(Values, b, tile, V: tl.constexpr, ITEMS: tl.constexpr):
    t = tl.arange(0, 128)
    dtype: tl.constexpr = Values.dtype.element_ty
    subtotal = tl.full([128], 0.0, tl.float32)
    for k in tl.static_range(ITEMS):
        idx = tile * 128 * ITEMS + t * ITEMS + k
        val = tl.load(Values + b * V + idx, idx < V, 0).to(tl.float32)
        subtotal = (subtotal + val).to(dtype).to(tl.float32)
    inclusive = subtotal
    for j in tl.static_range(5):
        offset = 1 << j
        prev = tl.gather(inclusive, tl.maximum(t - offset, 0), 0)
        inclusive = tl.where(
            t % 32 >= offset,
            (prev + inclusive).to(dtype).to(tl.float32),
            inclusive,
        )
    warp_prefix = tl.full([128], 0.0, tl.float32)
    total = tl.full((), 0.0, tl.float32)
    for warp in tl.static_range(4):
        warp_prefix = tl.where(t // 32 == warp, total, warp_prefix)
        wsum = tl.sum(tl.where(t == warp * 32 + 31, inclusive, 0.0), 0)
        total = (total + wsum).to(dtype).to(tl.float32)
    prefix = tl.where(
        t % 32 != 0, tl.gather(inclusive, tl.maximum(t - 1, 0), 0), 0.0
    )
    prefix = (warp_prefix + prefix).to(dtype).to(tl.float32)
    return (prefix, total)


@triton.jit
def base_tile_totals_kernel(
    Values, Totals, V: tl.constexpr, ITEMS: tl.constexpr, B: tl.constexpr
):
    blocks: tl.constexpr = (V + 128 * ITEMS - 1) // (128 * ITEMS)
    for work in range(tl.program_id(0), B * blocks, tl.num_programs(0)):
        b = work // blocks
        tile = work % blocks
        prefix, total = base_group_prefix(Values, b, tile, V, ITEMS)
        tl.store(Totals + b * tl.cdiv(V, 128 * ITEMS) + tile, total)


@triton.jit
def base_sample_kernel(
    Values,
    Totals,
    Sums,
    Coins,
    Predicts,
    Slots,
    V: tl.constexpr,
    ITEMS: tl.constexpr,
    LOW_THRESHOLD: tl.constexpr,
    B: tl.constexpr,
):
    blocks: tl.constexpr = (V + 128 * ITEMS - 1) // (128 * ITEMS)
    for work in range(tl.program_id(0), B * blocks, tl.num_programs(0)):
        b = work // blocks
        tile = work % blocks
        t = tl.arange(0, 128)
        dtype: tl.constexpr = Values.dtype.element_ty
        prefix, unused = base_group_prefix(Values, b, tile, V, ITEMS)
        carry = tl.full((), 0.0, tl.float32)
        for window in range(tl.cdiv(tile, 32)):
            previous = tile - 1 - window * 32 - t
            partial = tl.load(
                Totals + b * tl.cdiv(V, 128 * ITEMS) + previous,
                (previous >= 0) & (t < 32),
                0,
            ).to(tl.float32)
            for j in tl.static_range(5):
                other = tl.gather(partial, tl.minimum(t + (1 << j), 127), 0)
                partial = (partial + other).to(dtype).to(tl.float32)
            window_sum = tl.sum(tl.where(t == 0, partial, 0.0), 0)
            carry = (window_sum + carry).to(dtype).to(tl.float32)
        current = (carry + prefix).to(dtype).to(tl.float32)
        total = tl.load(Sums + b).to(tl.float32)
        coin = tl.load(Coins + b).to(tl.float32)
        threshold = coin * total
        if LOW_THRESHOLD:
            threshold = threshold.to(dtype).to(tl.float32)
        token = tl.full([128], V - 1, tl.int32)
        for k in tl.static_range(ITEMS):
            idx = tile * 128 * ITEMS + t * ITEMS + k
            val = tl.load(Values + b * V + idx, idx < V, 0).to(tl.float32)
            current = (current + val).to(dtype).to(tl.float32)
            token = tl.minimum(
                token, tl.where((idx < V) & (current > threshold), idx, V - 1)
            )
        selected = tl.min(token, 0)
        slot = tl.load(Slots + b)
        tl.atomic_min(Predicts + slot, selected, sem="relaxed")


def base_plain(
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
    source_i, cand_scale = base_as_index(source)
    index_i, idx_scale = base_as_index(index)
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
    base_accept_kernel[batch,](
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
    items = 15 if target.dtype == torch.float32 else 30
    tiles = triton.cdiv(vocab, 128 * items)
    totals = torch.empty(
        (batch, tiles), dtype=target.dtype, device=target.device
    )
    nt = min(512, 1 << (vocab // 4 if vocab > 128 else vocab).bit_length() - 1)
    base_values_kernel[min(batch * triton.cdiv(vocab, 256), 65535),](
        values,
        target,
        draft,
        cur_row,
        all_accepted,
        S=steps,
        V=vocab,
        FMAX=max_value,
        BLOCK=256,
        enable_fp_fusion=False,
        B=batch,
    )
    base_sum_kernel[batch,](
        values,
        sums,
        predicts_i,
        last_slot,
        vocab,
        nt,
        nt.bit_length() - 1,
        min(5, nt.bit_length() - 1),
        enable_fp_fusion=False,
    )
    base_tile_totals_kernel[min(batch * tiles, 65535),](
        values, totals, vocab, items, enable_fp_fusion=False, B=batch
    )
    base_sample_kernel[min(batch * tiles, 65535),](
        values,
        totals,
        sums,
        coins_final,
        predicts_i,
        last_slot,
        vocab,
        items,
        target.dtype == coins_final.dtype and target.dtype != torch.float32,
        enable_fp_fusion=False,
        B=batch,
    )
    if predicts_i is not predicts:
        predicts.copy_(predicts_i.to(predicts.dtype))
    if accept_i is not accept_index:
        accept_index.copy_(accept_i.to(accept_index.dtype))
    return (predicts, accept_index, accept_num)


def _isolated_sampling(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    if target_probs.dtype != torch.float32:
        return base_plain(
            candidates,
            retrive_index,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            num_slots,
        )
    return _vendor_plain(
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        num_slots,
    )


def _vendor_plain(
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
    source_i, cand_scale = base_as_index(source)
    index_i, idx_scale = base_as_index(index)
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
    base_accept_kernel[batch,](
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
    nt = min(512, 1 << (vocab // 4 if vocab > 128 else vocab).bit_length() - 1)
    base_values_kernel[min(batch * triton.cdiv(vocab, 256), 65535),](
        values,
        target,
        draft,
        cur_row,
        all_accepted,
        S=steps,
        V=vocab,
        FMAX=max_value,
        BLOCK=256,
        enable_fp_fusion=False,
        B=batch,
    )
    base_sum_kernel[batch,](
        values,
        sums,
        predicts_i,
        last_slot,
        vocab,
        nt,
        nt.bit_length() - 1,
        min(5, nt.bit_length() - 1),
        enable_fp_fusion=False,
    )
    sample_cdf(values, sums, coins_final, predicts_i, last_slot)
    if predicts_i is not predicts:
        predicts.copy_(predicts_i.to(predicts.dtype))
    if accept_i is not accept_index:
        accept_index.copy_(accept_i.to(accept_index.dtype))
    return (predicts, accept_index, accept_num)


@triton.jit
def _hillis_stage(
    Source, Destination, B: tl.constexpr, V: tl.constexpr, OFFSET: tl.constexpr
):
    tiles: tl.constexpr = (V + 255) // 256
    for program in range(tl.program_id(0), B * tiles, tl.num_programs(0)):
        row = program // tiles
        index = program % tiles * 256 + tl.arange(0, 256)
        current = tl.load(Source + row * V + index, index < V, 0).to(
            tl.float32
        )
        previous = tl.load(
            Source + row * V + index - OFFSET,
            (index >= OFFSET) & (index < V),
            0,
        ).to(tl.float32)
        result = tl.where(index >= OFFSET, previous + current, current)
        tl.store(Destination + row * V + index, result, index < V)


def hillis_cdf(values):
    batch, vocab = values.shape
    first = torch.empty_like(values)
    second = torch.empty_like(values)
    source = values
    destination = first
    for stage in range((vocab - 1).bit_length()):
        _hillis_stage[min(batch * triton.cdiv(vocab, 256), 65535),](
            source,
            destination,
            batch,
            vocab,
            1 << stage,
            num_warps=4,
            enable_fp_fusion=False,
        )
        source = destination
        destination = second if destination is first else first
    return source


@triton.jit
def _vendor_token_tiles(
    CDF,
    Sums,
    Coins,
    Tokens,
    B: tl.constexpr,
    V: tl.constexpr,
    TILES: tl.constexpr,
):
    for program in range(tl.program_id(0), B * TILES, tl.num_programs(0)):
        row = program // TILES
        index = program % TILES * 256 + tl.arange(0, 256)
        value = tl.load(CDF + row * V + index, index < V, 0)
        threshold = tl.load(Sums + row).to(tl.float32) * tl.load(
            Coins + row
        ).to(tl.float32)
        token = tl.min(
            tl.where((index < V) & (value > threshold), index, V - 1), 0
        )
        tl.store(Tokens + program, token)


@triton.jit
def _vendor_token_finish(
    Tokens,
    Predicts,
    Slots,
    TILES: tl.constexpr,
    WIDTH: tl.constexpr,
    V: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.arange(0, WIDTH)
    value = tl.load(Tokens + row * TILES + index, index < TILES, V - 1)
    token = tl.min(value, 0)
    tl.store(Predicts + tl.load(Slots + row), token)


def sample_cdf(values, sums, coins, predicts, slots):
    batch, vocab = values.shape
    cumulative = hillis_cdf(values)
    tiles = triton.cdiv(vocab, 256)
    tokens = torch.empty(
        (batch, tiles), dtype=torch.int32, device=values.device
    )
    _vendor_token_tiles[min(batch * tiles, 65535),](
        cumulative,
        sums,
        coins,
        tokens,
        batch,
        vocab,
        tiles,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _vendor_token_finish[batch,](
        tokens,
        predicts,
        slots,
        tiles,
        triton.next_power_of_2(tiles),
        vocab,
        num_warps=4,
    )


@triton.jit
def _ordered_positions(Positions, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    wide: tl.constexpr = (COUNT + BLOCK - 1) // BLOCK * BLOCK - 1 > 2147483647
    position_type: tl.constexpr = tl.int64 if wide else tl.int32
    pos = tl.program_id(0).to(position_type) * BLOCK + tl.arange(0, BLOCK).to(
        position_type
    )
    tl.store(Positions + pos, pos, pos < COUNT)


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
    return (
        predicts.to(candidates.dtype),
        accept_index.to(retrive_index.dtype),
        accepted,
    )


__all__ = ["chain_speculative_sampling"]
