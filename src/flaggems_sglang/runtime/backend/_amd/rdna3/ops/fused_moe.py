# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fused_moe for RDNA3 (gfx1100): the experts of a token-choice MoE layer,
gate/up projection, activation, down projection and the weighted sum over the
top-k experts.

Up to two (token, expert) pairs per expert run without sorting: one program
per pair and column block writes float32 gate/up sums, split along K when
there are few pairs, and one program per token and column block forms the
activations of the token's top-k pairs, projects them down and adds them up
with the routing weights.

More pairs are sorted by expert once, and both projections run as grouped
GEMMs on the matrix units over the sorted rows. The gate/up kernel applies
the activation and writes the intermediate as a 16-bit value plus its 16-bit
residual, which the down kernel multiplies with the weights in two dots, so
the result is as accurate as with a float32 intermediate. The down kernel
scales the rows by the routing weights in float32, and a last kernel adds up
the top-k rows of each token.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _activation(gate, ACT: tl.constexpr):
    if ACT == 0:
        return gate * tl.sigmoid(gate)
    return 0.5 * gate * (1.0 + tl.erf(gate * 0.7071067811865476))


# ---------------------------------------------------------------------------
# few pairs: no sorting
# ---------------------------------------------------------------------------
@triton.jit
def _pair_gate_up_kernel(
    x_ptr,
    w1_ptr,
    ids_ptr,
    part_ptr,
    P,
    H,
    inter,
    top_k,
    stride_xm,
    stride_we,
    stride_wn,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Writes the gate and up sums of one pair over every SPLITS-th K block,
    # part[split, pair, 0 | 1, :]; the down kernel adds the splits up.
    pair = tl.program_id(0)
    pid_n = tl.program_id(1)
    split = tl.program_id(2)
    token = pair // top_k
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < inter
    offs_k = tl.arange(0, BLOCK_K)
    x_row = x_ptr + token.to(tl.int64) * stride_xm
    w_gate = (
        w1_ptr
        + expert * stride_we
        + offs_n[None, :].to(tl.int64) * stride_wn
        + offs_k[:, None]
    )
    w_up = w_gate + inter * stride_wn
    acc_gate = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k in range(split * BLOCK_K, H, SPLITS * BLOCK_K):
        k_mask = k + offs_k < H
        a = tl.load(x_row + k + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        mask = k_mask[:, None] & n_mask[None, :]
        b_gate = tl.load(w_gate + k, mask=mask, other=0.0).to(tl.float32)
        b_up = tl.load(w_up + k, mask=mask, other=0.0).to(tl.float32)
        acc_gate += a[:, None] * b_gate
        acc_up += a[:, None] * b_up
    row = part_ptr + ((split * P + pair).to(tl.int64) * 2) * inter + offs_n
    tl.store(row, tl.sum(acc_gate, axis=0), mask=n_mask)
    tl.store(row + inter, tl.sum(acc_up, axis=0), mask=n_mask)


@triton.jit
def _token_down_kernel(
    part_ptr,
    w2_ptr,
    ids_ptr,
    weights_ptr,
    out_ptr,
    P,
    H,
    inter,
    top_k,
    scale,
    stride_we,
    stride_wn,
    stride_om,
    SPLITS: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The activation of each top-k pair is formed from its gate/up partial
    # sums and scaled by the routing weight; all products are summed
    # lane-wise and reduced once at the end.
    token = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < H
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for slot in range(0, top_k):
        pair = token * top_k + slot
        expert = tl.load(ids_ptr + pair).to(tl.int64)
        weight = tl.load(weights_ptr + pair).to(tl.float32)
        w_row = (
            w2_ptr
            + expert * stride_we
            + offs_n[None, :].to(tl.int64) * stride_wn
            + offs_k[:, None]
        )
        for k in range(0, inter, BLOCK_K):
            k_mask = k + offs_k < inter
            gate = tl.zeros((BLOCK_K,), dtype=tl.float32)
            up = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for split in tl.static_range(SPLITS):
                row = part_ptr + ((split * P + pair).to(tl.int64) * 2) * inter
                gate += tl.load(row + k + offs_k, mask=k_mask, other=0.0)
                up += tl.load(row + inter + k + offs_k, mask=k_mask, other=0.0)
            a = _activation(gate, ACT) * up * weight
            b = tl.load(
                w_row + k,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += a[:, None] * b
    tl.store(
        out_ptr + token.to(tl.int64) * stride_om + offs_n,
        (tl.sum(acc, axis=0) * scale).to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


# ---------------------------------------------------------------------------
# more pairs: sort by expert, grouped GEMMs
# ---------------------------------------------------------------------------
@triton.jit
def _moe_count_kernel(
    topk_ids_ptr,
    hist_ptr,
    P,
    E,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
):
    c = tl.program_id(0)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    ids = tl.load(topk_ids_ptr + offs, mask=offs < P, other=-1)
    valid = (ids >= 0) & (ids < E)
    ids = tl.where(valid, ids, 0)
    hist = tl.histogram(ids, BLOCK_E, mask=valid)
    tl.store(hist_ptr + c * BLOCK_E + tl.arange(0, BLOCK_E), hist)


@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    hist_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    P,
    E,
    num_chunks,
    num_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
    SUB: tl.constexpr,
    HIST_ROWS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # sorted_ids holds the pairs of each expert padded with P to a multiple
    # of BLOCK_M, block_expert the expert of each BLOCK_M block (>= E past
    # the last expert). Pairs with an expert id outside [0, E) are left out.
    # Program c < num_chunks places the pairs of chunk c, the others write
    # block_expert and the padding for BLOCK_B blocks each; every entry is
    # written once.
    c = tl.program_id(0)
    e_offs = tl.arange(0, BLOCK_E)
    counts = tl.zeros([BLOCK_E], dtype=tl.int32)
    before = tl.zeros([BLOCK_E], dtype=tl.int32)
    for r0 in range(0, num_chunks, HIST_ROWS):
        r = r0 + tl.arange(0, HIST_ROWS)
        hist = tl.load(
            hist_ptr + r[:, None] * BLOCK_E + e_offs[None, :],
            mask=(r < num_chunks)[:, None],
            other=0,
        )
        counts += tl.sum(hist, axis=0)
        before += tl.sum(tl.where((r < c)[:, None], hist, 0), axis=0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    pad_end = tl.cumsum(padded, axis=0)
    pad_start = pad_end - padded

    if c >= num_chunks:
        b = (c - num_chunks) * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b < num_blocks
        done = pad_end[None, :] <= b[:, None] * BLOCK_M
        expert = tl.sum(done.to(tl.int32), axis=1)
        tl.store(block_expert_ptr + b, expert, mask=b_mask)
        e_c = tl.minimum(expert, BLOCK_E - 1)
        fill_start = tl.gather(pad_start + counts, e_c, 0)
        slot = b[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
        used = b_mask & (expert < E)
        is_pad = used[:, None] & (slot >= fill_start[:, None])
        pad_id = tl.full([BLOCK_B, BLOCK_M], 0, tl.int32) + P
        tl.store(sorted_ids_ptr + slot, pad_id, mask=is_pad)
    else:
        run = pad_start + before
        local = tl.arange(0, SUB)
        for s in range(0, tl.minimum(P - c * CHUNK, CHUNK), SUB):
            offs = c * CHUNK + s + local
            ids = tl.load(topk_ids_ptr + offs, mask=offs < P, other=-1)
            valid = (ids >= 0) & (ids < E)
            ids = tl.where(valid, ids, 0)
            same = (ids[:, None] == ids[None, :]) & valid[None, :]
            earlier = same & (local[None, :] < local[:, None])
            slot = tl.gather(run, ids, 0) + tl.sum(earlier.to(tl.int32), 1)
            tl.store(sorted_ids_ptr + slot, offs, mask=valid)
            run += tl.histogram(ids, BLOCK_E, mask=valid)


@triton.jit
def _store_rows(ptrs, tile, mask):
    # Without this round trip the 16-bit store keeps the WMMA layout and
    # writes two bytes a lane.
    rows: tl.constexpr = tile.shape[0]
    cols: tl.constexpr = tile.shape[1]
    low, high = tl.split(tl.reshape(tile, (rows, cols // 2, 2)))
    tile = tl.reshape(tl.join(low, high), (rows, cols))
    tl.store(ptrs, tile.to(ptrs.dtype.element_ty), mask=mask)


@triton.jit
def _block_ids(num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    # Runs GROUP_M row blocks through all column blocks before moving on, so
    # that both the rows and the expert weights are reread from L2.
    pid = tl.program_id(0)
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_m
    pid_n = pid % in_group // group_m
    return pid_m, pid_n


@triton.jit
def _grouped_gate_up_kernel(
    x_ptr,
    w1_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    c1_ptr,
    lo_offset,
    P,
    E,
    H,
    inter,
    top_k,
    stride_xm,
    stride_we,
    stride_wn,
    num_pid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACT: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _block_ids(
        tl.num_programs(0) // num_pid_n, num_pid_n, GROUP_M
    )
    expert = tl.load(block_expert_ptr + pid_m)
    if expert >= E:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_m)
    valid = pair < P
    token = pair // top_k
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < inter
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = x_ptr + token[:, None].to(tl.int64) * stride_xm + offs_k[None, :]
    g_ptrs = (
        w1_ptr
        + expert.to(tl.int64) * stride_we
        + offs_n[None, :].to(tl.int64) * stride_wn
        + offs_k[:, None]
    )
    u_ptrs = g_ptrs + inter * stride_wn
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ones = tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float32)
    for k in range(0, H, BLOCK_K):
        k_mask = k + offs_k < H
        a_mask = valid[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        mask = k_mask[:, None] & n_mask[None, :]
        b_gate = tl.load(g_ptrs, mask=mask, other=0.0)
        b_up = tl.load(u_ptrs, mask=mask, other=0.0)
        # Each product goes through its own accumulator and is added with
        # tl.fma: WMMA truncates what it adds into a float32 accumulator,
        # and Triton folds a plain + into the dot.
        acc_gate = tl.fma(tl.dot(a, b_gate), ones, acc_gate)
        acc_up = tl.fma(tl.dot(a, b_up), ones, acc_up)
        a_ptrs += BLOCK_K
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K
    hidden = _activation(acc_gate, ACT) * acc_up
    c1_ptrs = c1_ptr + offs_m[:, None].to(tl.int64) * inter + offs_n[None, :]
    mask = valid[:, None] & n_mask[None, :]
    high = hidden.to(c1_ptr.dtype.element_ty).to(tl.float32)
    _store_rows(c1_ptrs, high, mask)
    _store_rows(c1_ptrs + lo_offset, hidden - high, mask)


@triton.jit
def _grouped_down_kernel(
    c1_ptr,
    lo_offset,
    w2_ptr,
    sorted_ids_ptr,
    block_expert_ptr,
    weights_ptr,
    c3_ptr,
    P,
    E,
    H,
    inter,
    stride_we,
    stride_wn,
    num_pid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _block_ids(
        tl.num_programs(0) // num_pid_n, num_pid_n, GROUP_M
    )
    expert = tl.load(block_expert_ptr + pid_m)
    if expert >= E:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_m)
    valid = pair < P
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < H
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = c1_ptr + offs_m[:, None].to(tl.int64) * inter + offs_k[None, :]
    b_ptrs = (
        w2_ptr
        + expert.to(tl.int64) * stride_we
        + offs_n[None, :].to(tl.int64) * stride_wn
        + offs_k[:, None]
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ones = tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float32)
    for k in range(0, inter, BLOCK_K):
        k_mask = k + offs_k < inter
        a_mask = valid[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        low = tl.load(a_ptrs + lo_offset, mask=a_mask, other=0.0)
        acc = tl.fma(tl.dot(a, b) + tl.dot(low, b), ones, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    weight = tl.load(weights_ptr + pair, mask=valid, other=0.0)
    scaled = acc * weight.to(tl.float32)[:, None]
    c3_ptrs = c3_ptr + pair[:, None].to(tl.int64) * H + offs_n[None, :]
    tl.store(c3_ptrs, scaled, mask=valid[:, None] & n_mask[None, :])


@triton.jit
def _combine_kernel(
    c3_ptr,
    out_ptr,
    T,
    H,
    scale,
    stride_om,
    TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (offs_t < T)[:, None] & (offs_h < H)[None, :]
    rows = offs_t.to(tl.int64)[:, None] * (TOP_K * H) + offs_h[None, :]
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        acc += tl.load(c3_ptr + rows + slot * H, mask=mask, other=0.0).to(
            tl.float32
        )
    tl.store(
        out_ptr + offs_t.to(tl.int64)[:, None] * stride_om + offs_h[None, :],
        (acc * scale).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _moe_align(topk_ids, E, block_m, chunk=256, sub=64, num_warps=4):
    P = topk_ids.numel()
    device = topk_ids.device
    block_e = triton.next_power_of_2(E)
    num_chunks = triton.cdiv(P, chunk)
    num_blocks = triton.cdiv(P + E * (block_m - 1), block_m)
    hist = torch.empty(num_chunks * block_e, dtype=torch.int32, device=device)
    sorted_ids = torch.empty(
        num_blocks * block_m, dtype=torch.int32, device=device
    )
    block_expert = torch.empty(num_blocks, dtype=torch.int32, device=device)
    _moe_count_kernel[(num_chunks,)](
        topk_ids, hist, P, E, BLOCK_E=block_e, CHUNK=chunk
    )
    block_b = max(1, min(4096 // block_e, 4096 // block_m))
    grid = (num_chunks + triton.cdiv(num_blocks, block_b),)
    _moe_align_kernel[grid](
        topk_ids,
        hist,
        sorted_ids,
        block_expert,
        P,
        E,
        num_chunks,
        num_blocks,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        CHUNK=chunk,
        SUB=sub,
        HIST_ROWS=max(1, 4096 // block_e),
        BLOCK_B=block_b,
        num_warps=num_warps,
    )
    return sorted_ids, block_expert, num_blocks


def _use_pairs(P, E):
    # Up to two pairs per expert most expert weights are read about once on
    # either path, and the sort, the intermediate rows and the extra launches
    # of the grouped path cost more than its matrix units save.
    return P <= 2 * E


# (BLOCK_N, BLOCK_K, num_warps) of the gate/up kernel by the number of K
# splits: each split covers its part of K in one or two blocks.
_PAIR_UP = {1: (8, 512, 8), 2: (8, 512, 4), 4: (64, 256, 8), 8: (64, 128, 8)}


def _pair_config(T, P, H, inter):
    """Splits of K, then (BLOCK_N, BLOCK_K, num_warps) of the gate/up and of
    the down kernel. Fewer than 512 gate/up programs leave the compute units
    idle, so K is split until there are about 1024."""
    programs = P * triton.cdiv(inter, 8)
    splits = 1
    if programs < 512:
        splits = min(8, triton.next_power_of_2(triton.cdiv(1024, programs)))
    while splits > 1 and splits * 256 > H:
        splits //= 2
    block_k = min(256, triton.next_power_of_2(inter))
    if inter > 256:
        down = (16, block_k, 8)
    else:
        down = (16 if T <= 2 else 64, block_k, 4)
    return splits, _PAIR_UP[splits], down


def _grouped_config(P, E, inter):
    """BLOCK_M, then (BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M) of
    the gate/up and of the down kernel."""
    if inter >= 512 and P >= 256 * E:
        return 64, (64, 64, 4, 2, 8), (256, 64, 8, 2, 8)
    if inter >= 512 and P >= 128 * E:
        return 32, (64, 64, 4, 2, 8), (256, 64, 8, 2, 8)
    return 32, (32, 128, 4, 2, 8), (256, 64, 8, 2, 8)


def _fused_moe_pairs(x, w1, w2, ids, weights, top_k, act, scale, out):
    T, H = x.shape
    inter = w2.shape[2]
    P = T * top_k
    splits, up_cfg, down_cfg = _pair_config(T, P, H, inter)
    part = torch.empty(
        (splits, P, 2, inter), dtype=torch.float32, device=x.device
    )
    _pair_gate_up_kernel[(P, triton.cdiv(inter, up_cfg[0]), splits)](
        x,
        w1,
        ids,
        part,
        P,
        H,
        inter,
        top_k,
        x.stride(0),
        w1.stride(0),
        w1.stride(1),
        SPLITS=splits,
        BLOCK_N=up_cfg[0],
        BLOCK_K=up_cfg[1],
        num_warps=up_cfg[2],
    )
    _token_down_kernel[(T, triton.cdiv(H, down_cfg[0]))](
        part,
        w2,
        ids,
        weights,
        out,
        P,
        H,
        inter,
        top_k,
        scale,
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        SPLITS=splits,
        ACT=act,
        BLOCK_N=down_cfg[0],
        BLOCK_K=down_cfg[1],
        num_warps=down_cfg[2],
    )
    return out


def _fused_moe_grouped(x, w1, w2, ids, weights, top_k, act, scale, out):
    T, H = x.shape
    E = w1.shape[0]
    inter = w2.shape[2]
    P = T * top_k
    block_m, up_cfg, down_cfg = _grouped_config(P, E, inter)
    sorted_ids, block_expert, num_blocks = _moe_align(ids, E, block_m)
    rows = num_blocks * block_m
    c1 = torch.empty((2, rows, inter), dtype=x.dtype, device=x.device)
    up_n = triton.cdiv(inter, up_cfg[0])
    _grouped_gate_up_kernel[(num_blocks * up_n,)](
        x,
        w1,
        sorted_ids,
        block_expert,
        c1,
        rows * inter,
        P,
        E,
        H,
        inter,
        top_k,
        x.stride(0),
        w1.stride(0),
        w1.stride(1),
        up_n,
        BLOCK_M=block_m,
        BLOCK_N=up_cfg[0],
        BLOCK_K=up_cfg[1],
        ACT=act,
        GROUP_M=up_cfg[4],
        num_warps=up_cfg[2],
        num_stages=up_cfg[3],
    )
    c3 = torch.empty((P, H), dtype=torch.float32, device=x.device)
    down_n = triton.cdiv(H, down_cfg[0])
    _grouped_down_kernel[(num_blocks * down_n,)](
        c1,
        rows * inter,
        w2,
        sorted_ids,
        block_expert,
        weights,
        c3,
        P,
        E,
        H,
        inter,
        w2.stride(0),
        w2.stride(1),
        down_n,
        BLOCK_M=block_m,
        BLOCK_N=down_cfg[0],
        BLOCK_K=down_cfg[1],
        GROUP_M=down_cfg[4],
        num_warps=down_cfg[2],
        num_stages=down_cfg[3],
    )
    _combine_kernel[(triton.cdiv(T, 4), triton.cdiv(H, 256))](
        c3,
        out,
        T,
        H,
        scale,
        out.stride(0),
        TOP_K=top_k,
        BLOCK_T=4,
        BLOCK_H=256,
        num_warps=4,
    )
    return out


def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_output,
    activation="silu",
    routed_scaling_factor=1.0,
):
    topk_weights, topk_ids, _ = topk_output
    if activation == "silu":
        act = 0
    elif activation == "gelu":
        act = 1
    else:
        raise ValueError("Unsupported activation: %s" % activation)
    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    out = torch.empty_like(hidden_states)
    if T == 0:
        return out
    x = (
        hidden_states
        if hidden_states.stride(1) == 1
        else hidden_states.contiguous()
    )
    w1 = w1 if w1.stride(2) == 1 else w1.contiguous()
    w2 = w2 if w2.stride(2) == 1 else w2.contiguous()
    ids = topk_ids.to(torch.int32).reshape(-1).contiguous()
    weights = topk_weights.reshape(-1).contiguous()
    args = (x, w1, w2, ids, weights, top_k, act, float(routed_scaling_factor))
    if _use_pairs(T * top_k, w1.shape[0]):
        return _fused_moe_pairs(*args, out)
    return _fused_moe_grouped(*args, out)


__all__ = ["fused_moe"]
