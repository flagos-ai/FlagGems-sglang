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


@triton.jit
def base_sum_partials(Values, Partials, V: tl.constexpr, NT: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1) * 128 + tl.arange(0, 128)
    a0 = tl.full([128], 0.0, tl.float32)
    a1 = tl.full([128], 0.0, tl.float32)
    a2 = tl.full([128], 0.0, tl.float32)
    a3 = tl.full([128], 0.0, tl.float32)
    for base in range(tl.cdiv(V, NT * 4)):
        if V > 128:
            idx = base * NT * 4 + t * 4
            delta: tl.constexpr = 1
            bound: tl.constexpr = V - V % 4
        else:
            idx = base * NT * 4 + t
            delta: tl.constexpr = NT
            bound: tl.constexpr = V
        a0 += tl.load(Values + b * V + idx, (idx < bound) & (t < NT), 0).to(
            tl.float32
        )
        a1 += tl.load(
            Values + b * V + idx + delta, (idx + delta < bound) & (t < NT), 0
        ).to(tl.float32)
        a2 += tl.load(
            Values + b * V + idx + 2 * delta,
            (idx + 2 * delta < bound) & (t < NT),
            0,
        ).to(tl.float32)
        a3 += tl.load(
            Values + b * V + idx + 3 * delta,
            (idx + 3 * delta < bound) & (t < NT),
            0,
        ).to(tl.float32)
    if V > 128:
        idx = V - V % 4 + t
        a0 += tl.load(Values + b * V + idx, (idx < V) & (t < NT), 0).to(
            tl.float32
        )
    result = a0 + a1 + a2 + a3
    tl.store(Partials + b * NT + t, result, t < NT)


@triton.jit
def base_sum_stage(
    In, Out, NT: tl.constexpr, ACTIVE: tl.constexpr, OFFSET: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1) * 128 + tl.arange(0, 128)
    a = tl.load(In + b * NT + t, t < ACTIVE, 0)
    other = tl.load(In + b * NT + t + OFFSET, t + OFFSET < ACTIVE, 0)
    tl.store(Out + b * NT + t, a + other, t < ACTIVE)


@triton.jit
def base_sum_finish(
    In, Sums, Predicts, Slots, NT: tl.constexpr, V: tl.constexpr
):
    b = tl.program_id(0)
    total = tl.load(In + b * NT)
    tl.store(Sums + b, total)
    slot = tl.load(Slots + b)
    tl.store(Predicts + slot, V - 1)


@triton.jit
def base_group_partials(Values, Groups, V: tl.constexpr, ITEMS: tl.constexpr):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    t = tl.arange(0, 128)
    dtype: tl.constexpr = Values.dtype.element_ty
    subtotal = tl.full([128], 0.0, tl.float32)
    for k in range(ITEMS):
        idx = tile * 128 * ITEMS + t * ITEMS + k
        val = tl.load(Values + b * V + idx, idx < V, 0).to(tl.float32)
        subtotal = (subtotal + val).to(dtype).to(tl.float32)
    tl.store(Groups + (b * tl.cdiv(V, 128 * ITEMS) + tile) * 128 + t, subtotal)


@triton.jit
def base_warp_scan_stage(In, Out, TILES: tl.constexpr, OFFSET: tl.constexpr):
    block = tl.program_id(0) * TILES + tl.program_id(1)
    t = tl.arange(0, 128)
    a = tl.load(In + block * 128 + t)
    previous = tl.load(In + block * 128 + t - OFFSET, t % 32 >= OFFSET, 0)
    tl.store(Out + block * 128 + t, a.to(tl.float32) + previous.to(tl.float32))


@triton.jit
def base_block_totals(Groups, Totals, TILES: tl.constexpr):
    block = tl.program_id(0) * TILES + tl.program_id(1)
    dtype: tl.constexpr = Groups.dtype.element_ty
    a = tl.load(Groups + block * 128 + 31).to(tl.float32)
    b = tl.load(Groups + block * 128 + 63).to(tl.float32)
    c = tl.load(Groups + block * 128 + 95).to(tl.float32)
    d = tl.load(Groups + block * 128 + 127).to(tl.float32)
    total = (
        ((a + b).to(dtype).to(tl.float32) + c).to(dtype).to(tl.float32) + d
    ).to(dtype)
    tl.store(Totals + block, total)


@triton.jit
def base_reverse_window(Totals, base, remaining, dtype: tl.constexpr):
    a0 = tl.load(Totals + base - 0, remaining > 0, 0).to(tl.float32)
    a1 = tl.load(Totals + base - 1, remaining > 1, 0).to(tl.float32)
    a2 = tl.load(Totals + base - 2, remaining > 2, 0).to(tl.float32)
    a3 = tl.load(Totals + base - 3, remaining > 3, 0).to(tl.float32)
    a4 = tl.load(Totals + base - 4, remaining > 4, 0).to(tl.float32)
    a5 = tl.load(Totals + base - 5, remaining > 5, 0).to(tl.float32)
    a6 = tl.load(Totals + base - 6, remaining > 6, 0).to(tl.float32)
    a7 = tl.load(Totals + base - 7, remaining > 7, 0).to(tl.float32)
    a8 = tl.load(Totals + base - 8, remaining > 8, 0).to(tl.float32)
    a9 = tl.load(Totals + base - 9, remaining > 9, 0).to(tl.float32)
    a10 = tl.load(Totals + base - 10, remaining > 10, 0).to(tl.float32)
    a11 = tl.load(Totals + base - 11, remaining > 11, 0).to(tl.float32)
    a12 = tl.load(Totals + base - 12, remaining > 12, 0).to(tl.float32)
    a13 = tl.load(Totals + base - 13, remaining > 13, 0).to(tl.float32)
    a14 = tl.load(Totals + base - 14, remaining > 14, 0).to(tl.float32)
    a15 = tl.load(Totals + base - 15, remaining > 15, 0).to(tl.float32)
    a16 = tl.load(Totals + base - 16, remaining > 16, 0).to(tl.float32)
    a17 = tl.load(Totals + base - 17, remaining > 17, 0).to(tl.float32)
    a18 = tl.load(Totals + base - 18, remaining > 18, 0).to(tl.float32)
    a19 = tl.load(Totals + base - 19, remaining > 19, 0).to(tl.float32)
    a20 = tl.load(Totals + base - 20, remaining > 20, 0).to(tl.float32)
    a21 = tl.load(Totals + base - 21, remaining > 21, 0).to(tl.float32)
    a22 = tl.load(Totals + base - 22, remaining > 22, 0).to(tl.float32)
    a23 = tl.load(Totals + base - 23, remaining > 23, 0).to(tl.float32)
    a24 = tl.load(Totals + base - 24, remaining > 24, 0).to(tl.float32)
    a25 = tl.load(Totals + base - 25, remaining > 25, 0).to(tl.float32)
    a26 = tl.load(Totals + base - 26, remaining > 26, 0).to(tl.float32)
    a27 = tl.load(Totals + base - 27, remaining > 27, 0).to(tl.float32)
    a28 = tl.load(Totals + base - 28, remaining > 28, 0).to(tl.float32)
    a29 = tl.load(Totals + base - 29, remaining > 29, 0).to(tl.float32)
    a30 = tl.load(Totals + base - 30, remaining > 30, 0).to(tl.float32)
    a31 = tl.load(Totals + base - 31, remaining > 31, 0).to(tl.float32)
    r0_0 = (a0 + a1).to(dtype).to(tl.float32)
    r0_1 = (a2 + a3).to(dtype).to(tl.float32)
    r0_2 = (a4 + a5).to(dtype).to(tl.float32)
    r0_3 = (a6 + a7).to(dtype).to(tl.float32)
    r0_4 = (a8 + a9).to(dtype).to(tl.float32)
    r0_5 = (a10 + a11).to(dtype).to(tl.float32)
    r0_6 = (a12 + a13).to(dtype).to(tl.float32)
    r0_7 = (a14 + a15).to(dtype).to(tl.float32)
    r0_8 = (a16 + a17).to(dtype).to(tl.float32)
    r0_9 = (a18 + a19).to(dtype).to(tl.float32)
    r0_10 = (a20 + a21).to(dtype).to(tl.float32)
    r0_11 = (a22 + a23).to(dtype).to(tl.float32)
    r0_12 = (a24 + a25).to(dtype).to(tl.float32)
    r0_13 = (a26 + a27).to(dtype).to(tl.float32)
    r0_14 = (a28 + a29).to(dtype).to(tl.float32)
    r0_15 = (a30 + a31).to(dtype).to(tl.float32)
    r1_0 = (r0_0 + r0_1).to(dtype).to(tl.float32)
    r1_1 = (r0_2 + r0_3).to(dtype).to(tl.float32)
    r1_2 = (r0_4 + r0_5).to(dtype).to(tl.float32)
    r1_3 = (r0_6 + r0_7).to(dtype).to(tl.float32)
    r1_4 = (r0_8 + r0_9).to(dtype).to(tl.float32)
    r1_5 = (r0_10 + r0_11).to(dtype).to(tl.float32)
    r1_6 = (r0_12 + r0_13).to(dtype).to(tl.float32)
    r1_7 = (r0_14 + r0_15).to(dtype).to(tl.float32)
    r2_0 = (r1_0 + r1_1).to(dtype).to(tl.float32)
    r2_1 = (r1_2 + r1_3).to(dtype).to(tl.float32)
    r2_2 = (r1_4 + r1_5).to(dtype).to(tl.float32)
    r2_3 = (r1_6 + r1_7).to(dtype).to(tl.float32)
    r3_0 = (r2_0 + r2_1).to(dtype).to(tl.float32)
    r3_1 = (r2_2 + r2_3).to(dtype).to(tl.float32)
    r4_0 = (r3_0 + r3_1).to(dtype).to(tl.float32)
    return r4_0


@triton.jit
def base_sample_prefix(
    Groups, Totals, Prefixes, V: tl.constexpr, ITEMS: tl.constexpr
):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    t = tl.arange(0, 128)
    dtype: tl.constexpr = Groups.dtype.element_ty
    tiles: tl.constexpr = tl.cdiv(V, 128 * ITEMS)
    base = (b * tiles + tile) * 128
    a = tl.load(Groups + base + 31).to(tl.float32)
    c = tl.load(Groups + base + 63).to(tl.float32)
    d = tl.load(Groups + base + 95).to(tl.float32)
    ab = (a + c).to(dtype).to(tl.float32)
    abc = (ab + d).to(dtype).to(tl.float32)
    warp_prefix = tl.where(
        t < 32, 0.0, tl.where(t < 64, a, tl.where(t < 96, ab, abc))
    )
    prefix = tl.load(Groups + base + t - 1, t % 32 != 0, 0).to(tl.float32)
    prefix = (warp_prefix + prefix).to(dtype).to(tl.float32)
    carry = tl.full((), 0.0, tl.float32)
    for window in range(tl.cdiv(tile, 32)):
        remaining = tile - window * 32
        part = base_reverse_window(
            Totals, b * tiles + remaining - 1, remaining, dtype
        )
        carry = (part + carry).to(dtype).to(tl.float32)
    current = (carry + prefix).to(dtype).to(tl.float32)
    tl.store(Prefixes + base + t, current)


@triton.jit
def base_sample_kernel(
    Values,
    Prefixes,
    Sums,
    Coins,
    Predicts,
    Slots,
    V: tl.constexpr,
    ITEMS: tl.constexpr,
    LOW_THRESHOLD: tl.constexpr,
):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    t = tl.arange(0, 128)
    dtype: tl.constexpr = Values.dtype.element_ty
    tiles: tl.constexpr = tl.cdiv(V, 128 * ITEMS)
    current = tl.load(Prefixes + (b * tiles + tile) * 128 + t).to(tl.float32)
    threshold = tl.load(Coins + b).to(tl.float32) * tl.load(Sums + b).to(
        tl.float32
    )
    if LOW_THRESHOLD:
        threshold = threshold.to(dtype).to(tl.float32)
    token = tl.full([128], V - 1, tl.int32)
    for k in range(ITEMS):
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
    partial = torch.empty(
        (batch, nt), dtype=torch.float32, device=target.device
    )
    partial_next = torch.empty_like(partial)
    groups = torch.empty(
        (batch, tiles, 128), dtype=target.dtype, device=target.device
    )
    groups_next = torch.empty_like(groups)
    base_values_kernel[batch, triton.cdiv(vocab, 128)](
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
    base_sum_partials[batch, triton.cdiv(nt, 128)](
        values, partial, vocab, nt, enable_fp_fusion=False
    )
    active = nt
    while active > 32:
        base_sum_stage[batch, triton.cdiv(active, 128)](
            partial,
            partial_next,
            nt,
            active,
            active // 2,
            enable_fp_fusion=False,
        )
        partial, partial_next = (partial_next, partial)
        active //= 2
    offset = 1
    while offset < active:
        base_sum_stage[batch, 1](
            partial, partial_next, nt, active, offset, enable_fp_fusion=False
        )
        partial, partial_next = (partial_next, partial)
        offset *= 2
    base_sum_finish[batch,](
        partial, sums, predicts_i, last_slot, nt, vocab, enable_fp_fusion=False
    )
    base_group_partials[batch, tiles](
        values, groups, vocab, items, enable_fp_fusion=False
    )
    for offset in (1, 2, 4, 8, 16):
        base_warp_scan_stage[batch, tiles](
            groups, groups_next, tiles, offset, enable_fp_fusion=False
        )
        groups, groups_next = (groups_next, groups)
    base_block_totals[batch, tiles](
        groups, totals, tiles, enable_fp_fusion=False
    )
    base_sample_prefix[batch, tiles](
        groups, totals, groups_next, vocab, items, enable_fp_fusion=False
    )
    base_sample_kernel[batch, tiles](
        values,
        groups_next,
        sums,
        coins_final,
        predicts_i,
        last_slot,
        vocab,
        items,
        target.dtype == coins_final.dtype and target.dtype != torch.float32,
        enable_fp_fusion=False,
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
    if target_probs.dtype != torch.float32 or target_probs.shape[-1] <= 4:
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
    base_values_kernel[batch, triton.cdiv(vocab, 128)](
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
    _maca_sum(values, sums, predicts_i, last_slot)
    sample_cdf(values, sums, coins_final, predicts_i, last_slot)
    if predicts_i is not predicts:
        predicts.copy_(predicts_i.to(predicts.dtype))
    if accept_i is not accept_index:
        accept_index.copy_(accept_i.to(accept_index.dtype))
    return (predicts, accept_index, accept_num)


@triton.jit
def _cub_partials(Values, Groups, B: tl.constexpr, V: tl.constexpr):
    tiles: tl.constexpr = (V + 895) // 896
    lane = tl.arange(0, 128)
    for program in range(tl.program_id(0), B * tiles, tl.num_programs(0)):
        row = program // tiles
        tile = program % tiles
        index = tile * 896 + lane * 7
        subtotal = tl.full([128], 0.0, tl.float32)
        for k in tl.static_range(7):
            subtotal = subtotal + tl.load(
                Values + row * V + index + k, index + k < V, 0
            ).to(tl.float32)
        tl.store(Groups + program * 128 + lane, subtotal)


@triton.jit
def _cub_group_stage(
    Source, Destination, COUNT: tl.constexpr, OFFSET: tl.constexpr
):
    lane = tl.arange(0, 128)
    for program in range(tl.program_id(0), COUNT, tl.num_programs(0)):
        current = tl.load(Source + program * 128 + lane)
        previous = tl.load(
            Source + program * 128 + lane - OFFSET, lane % 64 >= OFFSET, 0
        )
        result = tl.where(lane % 64 >= OFFSET, previous + current, current)
        tl.store(Destination + program * 128 + lane, result)


@triton.jit
def _cub_prefix(Groups, Prefix, Totals, COUNT: tl.constexpr):
    lane = tl.arange(0, 128)
    for program in range(tl.program_id(0), COUNT, tl.num_programs(0)):
        warp0 = tl.load(Groups + program * 128 + 63)
        warp1 = tl.load(Groups + program * 128 + 127)
        local = tl.load(Groups + program * 128 + lane - 1, lane % 64 != 0, 0)
        before = tl.where(lane >= 64, warp0, 0.0) + local
        tl.store(Prefix + program * 128 + lane, before)
        tl.store(Totals + program, warp0 + warp1)


@triton.jit
def _cub_carry(Totals, Carry, B: tl.constexpr, TILES: tl.constexpr):
    for program in range(tl.program_id(0), B * TILES, tl.num_programs(0)):
        row = program // TILES
        tile = program % TILES
        carry = tl.full((), 0.0, tl.float32)
        for window in range(tl.cdiv(tile, 64)):
            remaining = tile - window * 64
            first = base_reverse_window(
                Totals, row * TILES + remaining - 1, remaining, tl.float32
            )
            second = base_reverse_window(
                Totals,
                row * TILES + remaining - 33,
                remaining - 32,
                tl.float32,
            )
            carry = first + second + carry
        tl.store(Carry + program, carry)


@triton.jit
def _cub_write(Values, Prefix, Carry, CDF, B: tl.constexpr, V: tl.constexpr):
    tiles: tl.constexpr = (V + 895) // 896
    lane = tl.arange(0, 128)
    for program in range(tl.program_id(0), B * tiles, tl.num_programs(0)):
        row = program // tiles
        index = program % tiles * 896 + lane * 7
        prefix = tl.load(Prefix + program * 128 + lane)
        carry = tl.load(Carry + program)
        current = carry + prefix
        for k in tl.static_range(7):
            current = current + tl.load(
                Values + row * V + index + k, index + k < V, 0
            ).to(tl.float32)
            tl.store(CDF + row * V + index + k, current, index + k < V)


def cub_cdf(values):
    batch, vocab = values.shape
    tiles = triton.cdiv(vocab, 896)
    prefix = torch.empty(
        (batch, tiles, 128), device=values.device, dtype=torch.float32
    )
    groups = torch.empty_like(prefix)
    groups_next = torch.empty_like(prefix)
    totals = torch.empty(
        (batch, tiles), device=values.device, dtype=torch.float32
    )
    carry = torch.empty_like(totals)
    output = torch.empty_like(values)
    grid = (min(batch * tiles, 65535),)
    _cub_partials[grid](
        values, groups, batch, vocab, num_warps=4, enable_fp_fusion=False
    )
    for offset in (1, 2, 4, 8, 16, 32):
        _cub_group_stage[grid](
            groups,
            groups_next,
            batch * tiles,
            offset,
            num_warps=4,
            enable_fp_fusion=False,
        )
        groups, groups_next = (groups_next, groups)
    _cub_prefix[grid](
        groups,
        prefix,
        totals,
        batch * tiles,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _cub_carry[grid](
        totals, carry, batch, tiles, num_warps=4, enable_fp_fusion=False
    )
    _cub_write[grid](
        values,
        prefix,
        carry,
        output,
        batch,
        vocab,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


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
    cumulative = cub_cdf(values)
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
def _maca_partials(
    Values,
    Partial,
    V: tl.constexpr,
    NT: tl.constexpr,
    VEC: tl.constexpr,
    CTAS: tl.constexpr,
):
    rc = tl.program_id(0)
    row = rc // CTAS
    cta = rc % CTAS
    lane = tl.program_id(1) * 128 + tl.arange(0, 128)
    current = tl.full([128], 0.0, tl.float32)
    for base in range(tl.cdiv(V, CTAS * NT * VEC)):
        start = ((base * CTAS + cta) * NT + lane) * VEC
        for k in tl.static_range(VEC):
            value = tl.load(
                Values + row * V + start + k, (lane < NT) & (start + k < V), 0
            ).to(tl.float32)
            current = current + value
    tl.store(Partial + rc * NT + lane, current, lane < NT)


@triton.jit
def _maca_global(Partial, Reduced, NT: tl.constexpr, CTAS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 64)
    current = tl.full([64], 0.0, tl.float32)
    for base in range(tl.cdiv(CTAS, 64)):
        index = base * 64 + lane
        value = tl.load(Partial + (row * CTAS + index) * NT, index < CTAS, 0)
        current += value
    tl.store(Reduced + row * 64 + lane, current)


def _maca_sum(values, sums, predicts, slots):
    batch, vocab = values.shape
    vector = 4 if vocab > 128 else 1
    while vocab % vector:
        vector //= 2
    nt = min(512, 1 << (vocab // vector).bit_length() - 1)
    ctas = 1
    partial = torch.empty(
        (batch * ctas, nt), dtype=torch.float32, device=values.device
    )
    other = torch.empty_like(partial)
    _maca_partials[batch * ctas, triton.cdiv(nt, 128)](
        values, partial, vocab, nt, vector, ctas, enable_fp_fusion=False
    )
    active = nt
    while active > 64:
        base_sum_stage[batch * ctas, triton.cdiv(active, 128)](
            partial, other, nt, active, active // 2, enable_fp_fusion=False
        )
        partial, other = (other, partial)
        active //= 2
    offset = 1
    while offset < active:
        base_sum_stage[batch * ctas, 1](
            partial, other, nt, active, offset, enable_fp_fusion=False
        )
        partial, other = (other, partial)
        offset *= 2
    if ctas > 1:
        reduced = torch.empty(
            (batch, 64), dtype=torch.float32, device=values.device
        )
        other = torch.empty_like(reduced)
        _maca_global[batch,](
            partial, reduced, nt, ctas, enable_fp_fusion=False
        )
        partial = reduced
        nt = 64
        offset = 1
        while offset < 64:
            base_sum_stage[batch, 1](
                partial, other, nt, 64, offset, enable_fp_fusion=False
            )
            partial, other = (other, partial)
            offset *= 2
    base_sum_finish[batch,](
        partial, sums, predicts, slots, nt, vocab, enable_fp_fusion=False
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
