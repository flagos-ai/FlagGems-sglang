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


def safe_as_index(tensor):
    if tensor.dtype != torch.int64:
        return (tensor, 1)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return (tensor.view(torch.int32), 2)


@triton.jit
def safe_accept_kernel(
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
def safe_values_kernel(
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
    p = tl.where(
        idx < V,
        tl.load(tl.where(idx < V, Target + (b * S + row) * V + idx, Target)),
        0,
    )
    if S > 1:
        q = tl.where(
            (idx < V) & (accepted == 0),
            tl.load(
                tl.where(
                    (idx < V) & (accepted == 0),
                    Draft + (b * (S - 1) + row) * V + idx,
                    Draft,
                )
            ),
            0,
        ).to(tl.float32)
    else:
        q = tl.full([128], 0.0, tl.float32)
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
def safe_sum_partials(Values, Partials, V: tl.constexpr, NT: tl.constexpr):
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
        a0 += tl.where(
            (idx < bound) & (t < NT),
            tl.load(
                tl.where(
                    (idx < bound) & (t < NT), Values + b * V + idx, Values
                )
            ),
            0,
        ).to(tl.float32)
        a1 += tl.where(
            (idx + delta < bound) & (t < NT),
            tl.load(
                tl.where(
                    (idx + delta < bound) & (t < NT),
                    Values + b * V + idx + delta,
                    Values,
                )
            ),
            0,
        ).to(tl.float32)
        a2 += tl.where(
            (idx + 2 * delta < bound) & (t < NT),
            tl.load(
                tl.where(
                    (idx + 2 * delta < bound) & (t < NT),
                    Values + b * V + idx + 2 * delta,
                    Values,
                )
            ),
            0,
        ).to(tl.float32)
        a3 += tl.where(
            (idx + 3 * delta < bound) & (t < NT),
            tl.load(
                tl.where(
                    (idx + 3 * delta < bound) & (t < NT),
                    Values + b * V + idx + 3 * delta,
                    Values,
                )
            ),
            0,
        ).to(tl.float32)
    if V > 128:
        idx = V - V % 4 + t
        a0 += tl.where(
            (idx < V) & (t < NT),
            tl.load(
                tl.where((idx < V) & (t < NT), Values + b * V + idx, Values)
            ),
            0,
        ).to(tl.float32)
    result = a0 + a1 + a2 + a3
    tl.store(Partials + b * NT + t, result, t < NT)


@triton.jit
def safe_sum_stage(
    In, Out, NT: tl.constexpr, ACTIVE: tl.constexpr, OFFSET: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1) * 128 + tl.arange(0, 128)
    a = tl.where(
        t < ACTIVE, tl.load(tl.where(t < ACTIVE, In + b * NT + t, In)), 0
    )
    other = tl.where(
        t + OFFSET < ACTIVE,
        tl.load(tl.where(t + OFFSET < ACTIVE, In + b * NT + t + OFFSET, In)),
        0,
    )
    tl.store(Out + b * NT + t, a + other, t < ACTIVE)


@triton.jit
def safe_sum_finish(
    In, Sums, Predicts, Slots, NT: tl.constexpr, V: tl.constexpr
):
    b = tl.program_id(0)
    total = tl.load(In + b * NT)
    tl.store(Sums + b, total)
    slot = tl.load(Slots + b)
    tl.store(Predicts + slot, V - 1)


@triton.jit
def safe_group_partials(Values, Groups, V: tl.constexpr, ITEMS: tl.constexpr):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    t = tl.arange(0, 128)
    dtype: tl.constexpr = Values.dtype.element_ty
    subtotal = tl.full([128], 0.0, tl.float32)
    for k in range(ITEMS):
        idx = tile * 128 * ITEMS + t * ITEMS + k
        val = tl.where(
            idx < V,
            tl.load(tl.where(idx < V, Values + b * V + idx, Values)),
            0,
        ).to(tl.float32)
        subtotal = (subtotal + val).to(dtype).to(tl.float32)
    tl.store(Groups + (b * tl.cdiv(V, 128 * ITEMS) + tile) * 128 + t, subtotal)


@triton.jit
def safe_warp_scan_stage(In, Out, TILES: tl.constexpr, OFFSET: tl.constexpr):
    block = tl.program_id(0) * TILES + tl.program_id(1)
    t = tl.arange(0, 128)
    a = tl.load(In + block * 128 + t)
    previous = tl.where(
        t % 32 >= OFFSET,
        tl.load(tl.where(t % 32 >= OFFSET, In + block * 128 + t - OFFSET, In)),
        0,
    )
    tl.store(Out + block * 128 + t, a.to(tl.float32) + previous.to(tl.float32))


@triton.jit
def safe_block_totals(Groups, Totals, TILES: tl.constexpr):
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
def safe_reverse_window(Totals, base, remaining, dtype: tl.constexpr):
    a0 = tl.where(
        remaining > 0,
        tl.load(tl.where(remaining > 0, Totals + base - 0, Totals)),
        0,
    ).to(tl.float32)
    a1 = tl.where(
        remaining > 1,
        tl.load(tl.where(remaining > 1, Totals + base - 1, Totals)),
        0,
    ).to(tl.float32)
    a2 = tl.where(
        remaining > 2,
        tl.load(tl.where(remaining > 2, Totals + base - 2, Totals)),
        0,
    ).to(tl.float32)
    a3 = tl.where(
        remaining > 3,
        tl.load(tl.where(remaining > 3, Totals + base - 3, Totals)),
        0,
    ).to(tl.float32)
    a4 = tl.where(
        remaining > 4,
        tl.load(tl.where(remaining > 4, Totals + base - 4, Totals)),
        0,
    ).to(tl.float32)
    a5 = tl.where(
        remaining > 5,
        tl.load(tl.where(remaining > 5, Totals + base - 5, Totals)),
        0,
    ).to(tl.float32)
    a6 = tl.where(
        remaining > 6,
        tl.load(tl.where(remaining > 6, Totals + base - 6, Totals)),
        0,
    ).to(tl.float32)
    a7 = tl.where(
        remaining > 7,
        tl.load(tl.where(remaining > 7, Totals + base - 7, Totals)),
        0,
    ).to(tl.float32)
    a8 = tl.where(
        remaining > 8,
        tl.load(tl.where(remaining > 8, Totals + base - 8, Totals)),
        0,
    ).to(tl.float32)
    a9 = tl.where(
        remaining > 9,
        tl.load(tl.where(remaining > 9, Totals + base - 9, Totals)),
        0,
    ).to(tl.float32)
    a10 = tl.where(
        remaining > 10,
        tl.load(tl.where(remaining > 10, Totals + base - 10, Totals)),
        0,
    ).to(tl.float32)
    a11 = tl.where(
        remaining > 11,
        tl.load(tl.where(remaining > 11, Totals + base - 11, Totals)),
        0,
    ).to(tl.float32)
    a12 = tl.where(
        remaining > 12,
        tl.load(tl.where(remaining > 12, Totals + base - 12, Totals)),
        0,
    ).to(tl.float32)
    a13 = tl.where(
        remaining > 13,
        tl.load(tl.where(remaining > 13, Totals + base - 13, Totals)),
        0,
    ).to(tl.float32)
    a14 = tl.where(
        remaining > 14,
        tl.load(tl.where(remaining > 14, Totals + base - 14, Totals)),
        0,
    ).to(tl.float32)
    a15 = tl.where(
        remaining > 15,
        tl.load(tl.where(remaining > 15, Totals + base - 15, Totals)),
        0,
    ).to(tl.float32)
    a16 = tl.where(
        remaining > 16,
        tl.load(tl.where(remaining > 16, Totals + base - 16, Totals)),
        0,
    ).to(tl.float32)
    a17 = tl.where(
        remaining > 17,
        tl.load(tl.where(remaining > 17, Totals + base - 17, Totals)),
        0,
    ).to(tl.float32)
    a18 = tl.where(
        remaining > 18,
        tl.load(tl.where(remaining > 18, Totals + base - 18, Totals)),
        0,
    ).to(tl.float32)
    a19 = tl.where(
        remaining > 19,
        tl.load(tl.where(remaining > 19, Totals + base - 19, Totals)),
        0,
    ).to(tl.float32)
    a20 = tl.where(
        remaining > 20,
        tl.load(tl.where(remaining > 20, Totals + base - 20, Totals)),
        0,
    ).to(tl.float32)
    a21 = tl.where(
        remaining > 21,
        tl.load(tl.where(remaining > 21, Totals + base - 21, Totals)),
        0,
    ).to(tl.float32)
    a22 = tl.where(
        remaining > 22,
        tl.load(tl.where(remaining > 22, Totals + base - 22, Totals)),
        0,
    ).to(tl.float32)
    a23 = tl.where(
        remaining > 23,
        tl.load(tl.where(remaining > 23, Totals + base - 23, Totals)),
        0,
    ).to(tl.float32)
    a24 = tl.where(
        remaining > 24,
        tl.load(tl.where(remaining > 24, Totals + base - 24, Totals)),
        0,
    ).to(tl.float32)
    a25 = tl.where(
        remaining > 25,
        tl.load(tl.where(remaining > 25, Totals + base - 25, Totals)),
        0,
    ).to(tl.float32)
    a26 = tl.where(
        remaining > 26,
        tl.load(tl.where(remaining > 26, Totals + base - 26, Totals)),
        0,
    ).to(tl.float32)
    a27 = tl.where(
        remaining > 27,
        tl.load(tl.where(remaining > 27, Totals + base - 27, Totals)),
        0,
    ).to(tl.float32)
    a28 = tl.where(
        remaining > 28,
        tl.load(tl.where(remaining > 28, Totals + base - 28, Totals)),
        0,
    ).to(tl.float32)
    a29 = tl.where(
        remaining > 29,
        tl.load(tl.where(remaining > 29, Totals + base - 29, Totals)),
        0,
    ).to(tl.float32)
    a30 = tl.where(
        remaining > 30,
        tl.load(tl.where(remaining > 30, Totals + base - 30, Totals)),
        0,
    ).to(tl.float32)
    a31 = tl.where(
        remaining > 31,
        tl.load(tl.where(remaining > 31, Totals + base - 31, Totals)),
        0,
    ).to(tl.float32)
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
def safe_sample_prefix(
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
    prefix = tl.where(
        t % 32 != 0,
        tl.load(tl.where(t % 32 != 0, Groups + base + t - 1, Groups)),
        0,
    ).to(tl.float32)
    prefix = (warp_prefix + prefix).to(dtype).to(tl.float32)
    carry = tl.full((), 0.0, tl.float32)
    for window in range(tl.cdiv(tile, 32)):
        remaining = tile - window * 32
        part = safe_reverse_window(
            Totals, b * tiles + remaining - 1, remaining, dtype
        )
        carry = (part + carry).to(dtype).to(tl.float32)
    current = (carry + prefix).to(dtype).to(tl.float32)
    tl.store(Prefixes + base + t, current)


@triton.jit(do_not_specialize=["K"])
def safe_sample_step(
    Values,
    Prefixes,
    Tokens,
    Sums,
    Coins,
    K,
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
    base = (b * tiles + tile) * 128 + t
    token = tl.load(Tokens + base)
    idx = tile * 128 * ITEMS + t * ITEMS + K
    val = tl.where(
        idx < V, tl.load(tl.where(idx < V, Values + b * V + idx, Values)), 0
    ).to(tl.float32)
    current = (current + val).to(dtype).to(tl.float32)
    token = tl.minimum(
        token, tl.where((idx < V) & (current > threshold), idx, V - 1)
    )
    tl.store(Prefixes + base, current)
    tl.store(Tokens + base, token)


@triton.jit
def writer_sample_finish(Tokens, Predicts, Slots, TILES: tl.constexpr):
    b = tl.program_id(0)
    t = tl.arange(0, 128)
    selected = tl.full((), 2147483647, tl.int32)
    for tile in range(TILES):
        token = tl.load(Tokens + (b * TILES + tile) * 128 + t)
        selected = tl.minimum(selected, tl.min(token, 0))
    slot = tl.load(Slots + b)
    tl.store(Predicts + slot, selected)


def writer_plain(
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
    source_i, cand_scale = safe_as_index(source)
    index_i, idx_scale = safe_as_index(index)
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
    safe_accept_kernel[batch,](
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
    tokens = torch.full(
        (batch, tiles, 128), vocab - 1, dtype=torch.int32, device=target.device
    )
    safe_values_kernel[batch, triton.cdiv(vocab, 128)](
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
    safe_sum_partials[batch, triton.cdiv(nt, 128)](
        values, partial, vocab, nt, enable_fp_fusion=False
    )
    active = nt
    while active > 32:
        safe_sum_stage[batch, triton.cdiv(active, 128)](
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
        safe_sum_stage[batch, 1](
            partial, partial_next, nt, active, offset, enable_fp_fusion=False
        )
        partial, partial_next = (partial_next, partial)
        offset *= 2
    safe_sum_finish[batch,](
        partial, sums, predicts_i, last_slot, nt, vocab, enable_fp_fusion=False
    )
    safe_group_partials[batch, tiles](
        values, groups, vocab, items, enable_fp_fusion=False
    )
    for offset in (1, 2, 4, 8, 16):
        safe_warp_scan_stage[batch, tiles](
            groups, groups_next, tiles, offset, enable_fp_fusion=False
        )
        groups, groups_next = (groups_next, groups)
    safe_block_totals[batch, tiles](
        groups, totals, tiles, enable_fp_fusion=False
    )
    safe_sample_prefix[batch, tiles](
        groups, totals, groups_next, vocab, items, enable_fp_fusion=False
    )
    for k in range(items):
        safe_sample_step[batch, tiles](
            values,
            groups_next,
            tokens,
            sums,
            coins_final,
            k,
            vocab,
            items,
            target.dtype == coins_final.dtype
            and target.dtype != torch.float32,
            enable_fp_fusion=False,
        )
    writer_sample_finish[batch,](
        tokens, predicts_i, last_slot, tiles, enable_fp_fusion=False
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
        return writer_plain(
            candidates,
            retrive_index,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            num_slots,
        )
    return _serial_plain(
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        num_slots,
    )


def _serial_plain(
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
    source_i, cand_scale = safe_as_index(source)
    index_i, idx_scale = safe_as_index(index)
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
    safe_accept_kernel[batch,](
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
    safe_values_kernel[batch, triton.cdiv(vocab, 128)](
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
    if vocab > 128:
        _segmented_sum(values, sums, predicts_i, last_slot)
    else:
        nt = min(
            512, 1 << (vocab // 4 if vocab > 128 else vocab).bit_length() - 1
        )
        partial = torch.empty(
            (batch, nt), dtype=torch.float32, device=target.device
        )
        partial_next = torch.empty_like(partial)
        safe_sum_partials[batch, triton.cdiv(nt, 128)](
            values, partial, vocab, nt, enable_fp_fusion=False
        )
        active = nt
        while active > 32:
            safe_sum_stage[batch, triton.cdiv(active, 128)](
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
            safe_sum_stage[batch, 1](
                partial,
                partial_next,
                nt,
                active,
                offset,
                enable_fp_fusion=False,
            )
            partial, partial_next = (partial_next, partial)
            offset *= 2
        safe_sum_finish[batch,](
            partial,
            sums,
            predicts_i,
            last_slot,
            nt,
            vocab,
            enable_fp_fusion=False,
        )
    _serial_sample[batch,](
        values,
        sums,
        coins_final,
        predicts_i,
        last_slot,
        vocab,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if predicts_i is not predicts:
        predicts.copy_(predicts_i.to(predicts.dtype))
    if accept_i is not accept_index:
        accept_index.copy_(accept_i.to(accept_index.dtype))
    return (predicts, accept_index, accept_num)


@triton.jit
def _serial_sample(Values, Sums, Coins, Predicts, Slots, V: tl.constexpr):
    row = tl.program_id(0)
    threshold = tl.load(Sums + row).to(tl.float32) * tl.load(Coins + row).to(
        tl.float32
    )
    cumulative = tl.full((), 0.0, tl.float32)
    token = tl.full((), V - 1, tl.int32)
    index = 0
    found = tl.full((), False, tl.int1)
    while (index < V) & (found == 0):
        value = tl.load(Values + row * V + index).to(tl.float32)
        cumulative = cumulative + value
        found = cumulative > threshold
        token = tl.where(found, index, token)
        index += 1
    slot = tl.load(Slots + row)
    tl.store(Predicts + slot, token)


@triton.jit
def _scalar_parts(
    Values, Parts, V: tl.constexpr, WIDTH: tl.constexpr, COUNT: tl.constexpr
):
    row = tl.program_id(0)
    for part in range(COUNT):
        base = part * WIDTH
        size = tl.minimum(WIDTH, V - base)
        entries = tl.cdiv(size, 16)
        start = tl.maximum(entries - 2, 0)
        total = tl.full((), 0.0, tl.float32)
        for lane in range(16):
            value = tl.full((), 0.0, tl.float32)
            for index in range(start, entries):
                offset = index * 16 + lane
                loaded = tl.load(
                    tl.where(
                        offset < size, Values + row * V + base + offset, Values
                    )
                )
                value += tl.where(offset < size, loaded, 0).to(tl.float32)
            for index in range(start):
                value += tl.load(
                    Values + row * V + base + index * 16 + lane
                ).to(tl.float32)
            total += value
        tl.store(Parts + row * COUNT + part, total)


@triton.jit
def _scalar_merge_stage(
    In, Out, COUNT: tl.constexpr, ACTIVE: tl.constexpr, GROUP: tl.constexpr
):
    row = tl.program_id(0)
    half: tl.constexpr = ACTIVE // 2
    for lane in range(half):
        if GROUP:
            left = lane // 16 * 32 + lane % 16
            right = left + 16
        else:
            left = lane
            right = lane + half
        a = tl.where(
            left < COUNT,
            tl.load(tl.where(left < COUNT, In + row * COUNT + left, In)),
            0,
        )
        b = tl.where(
            right < COUNT,
            tl.load(tl.where(right < COUNT, In + row * COUNT + right, In)),
            0,
        )
        tl.store(Out + row * half + lane, a + b)


def _segmented_sum(values, sums, predicts, slots):
    batch, vocab = values.shape
    width = max(16, (vocab // 64 + 31) // 32 * 32)
    count = (vocab + width - 1) // width
    active = max(16, triton.next_power_of_2(count))
    parts = torch.empty(
        (batch, count), device=values.device, dtype=torch.float32
    )
    _scalar_parts[batch,](
        values, parts, vocab, width, count, num_warps=4, enable_fp_fusion=False
    )
    current = parts
    live = count
    while active > 1:
        other = torch.empty(
            (batch, active // 2), device=values.device, dtype=torch.float32
        )
        _scalar_merge_stage[batch,](
            current,
            other,
            live,
            active,
            active > 16,
            num_warps=4,
            enable_fp_fusion=False,
        )
        current = other
        active //= 2
        live = active
    safe_sum_finish[batch,](
        current,
        sums,
        predicts,
        slots,
        1,
        vocab,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return parts


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
