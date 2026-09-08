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

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_PERSISTENT_PROGRAMS = 256
_PERSISTENT_WARPS = 8
_BLOCK_N_CAP = 64
_BLOCK_K_CAP = 64


@triton.jit
def _chunk_state_varlen_cube_persistent_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    da_ptr,
    cu_ptr,
    out_ptr,
    STRIDE_X_SEQLEN: tl.constexpr,
    STRIDE_X_HEAD: tl.constexpr,
    STRIDE_X_HDIM: tl.constexpr,
    STRIDE_B_SEQLEN: tl.constexpr,
    STRIDE_B_GROUP: tl.constexpr,
    STRIDE_B_DSTATE: tl.constexpr,
    STRIDE_DT_HEAD: tl.constexpr,
    STRIDE_DT_CHUNK: tl.constexpr,
    STRIDE_DT_CSIZE: tl.constexpr,
    STRIDE_DA_HEAD: tl.constexpr,
    STRIDE_DA_CHUNK: tl.constexpr,
    STRIDE_DA_CSIZE: tl.constexpr,
    STRIDE_OUT_BATCH: tl.constexpr,
    STRIDE_OUT_HEAD: tl.constexpr,
    STRIDE_OUT_HDIM: tl.constexpr,
    STRIDE_OUT_DSTATE: tl.constexpr,
    TOTAL_SEQLEN: tl.constexpr,
    NUM_TASKS: tl.constexpr,
    NCHUNKS: tl.constexpr,
    NGROUPS: tl.constexpr,
    RATIO: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM: tl.constexpr,
    DSTATE: tl.constexpr,
    HEADS_PER_TILE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m: tl.constexpr = HEADS_PER_TILE * BLOCK_P
    num_pid_n: tl.constexpr = tl.cdiv(DSTATE, BLOCK_N)
    num_pid_p: tl.constexpr = tl.cdiv(HEADDIM, BLOCK_P)
    num_pid_h: tl.constexpr = tl.cdiv(RATIO, HEADS_PER_TILE)
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    head_base = tl.arange(0, HEADS_PER_TILE)
    p_base = tl.arange(0, BLOCK_P)
    n_base = tl.arange(0, BLOCK_N)
    k_base = tl.arange(0, BLOCK_K)

    for task in tl.range(program_id, NUM_TASKS, num_programs):
        task_left = task
        pid_n = task_left % num_pid_n
        task_left = task_left // num_pid_n
        pid_p = task_left % num_pid_p
        task_left = task_left // num_pid_p
        pid_head_tile = task_left % num_pid_h
        task_left = task_left // num_pid_h
        pid_group = task_left % NGROUPS
        pid_batch = task_left // NGROUPS

        head_local = pid_head_tile * HEADS_PER_TILE + head_base
        head_ok = head_local < RATIO
        offs_head = pid_group * RATIO + head_local
        safe_head = offs_head
        offs_p = pid_p * BLOCK_P + p_base
        offs_n = pid_n * BLOCK_N + n_base
        p_ok = offs_p < HEADDIM
        n_ok = offs_n < DSTATE
        safe_p = offs_p
        safe_n = offs_n

        end_idx = tl.load(cu_ptr + pid_batch + 1)
        start_idx = tl.load(cu_ptr + pid_batch)
        pid_chunk = tl.maximum((end_idx - 1) // CHUNK_SIZE, 0)
        chunk_base = pid_chunk * CHUNK_SIZE
        seg_end = tl.minimum(tl.maximum(end_idx - chunk_base, 0), CHUNK_SIZE)
        seg_start = tl.minimum(tl.maximum(start_idx - chunk_base, 0), seg_end)
        dt_base = dt_ptr + pid_chunk * STRIDE_DT_CHUNK
        da_base = da_ptr + pid_chunk * STRIDE_DA_CHUNK
        da_last = tl.load(
            da_base
            + safe_head * STRIDE_DA_HEAD
            + tl.maximum(seg_end - 1, 0) * STRIDE_DA_CSIZE
        ).to(tl.float32)

        acc = tl.zeros((block_m, BLOCK_N), dtype=tl.float32)
        for k_block in tl.range(0, tl.cdiv(CHUNK_SIZE, BLOCK_K)):
            k_global = k_block * BLOCK_K + k_base
            valid = (k_global >= seg_start) & (k_global < seg_end)
            token = chunk_base + k_global

            x = tl.load(
                x_ptr
                + token[None, None, :] * STRIDE_X_SEQLEN
                + safe_head[:, None, None] * STRIDE_X_HEAD
                + safe_p[None, :, None] * STRIDE_X_HDIM
            ).to(tl.float16)
            b = tl.load(
                b_ptr
                + token[:, None] * STRIDE_B_SEQLEN
                + pid_group * STRIDE_B_GROUP
                + safe_n[None, :] * STRIDE_B_DSTATE
            ).to(tl.float16)
            dt = tl.load(
                dt_base
                + safe_head[:, None] * STRIDE_DT_HEAD
                + k_global[None, :] * STRIDE_DT_CSIZE
            ).to(tl.float32)
            da = tl.load(
                da_base
                + safe_head[:, None] * STRIDE_DA_HEAD
                + k_global[None, :] * STRIDE_DA_CSIZE
            ).to(tl.float32)
            scale = tl.where(
                head_ok[:, None] & valid[None, :],
                tl.exp(da_last[:, None] - da) * dt,
                0.0,
            ).to(tl.float16)
            a = (x * scale[:, None, :]).reshape((block_m, BLOCK_K))
            acc = tl.dot(a, b, acc)

        acc = acc.reshape((HEADS_PER_TILE, BLOCK_P, BLOCK_N))
        out_ptrs = (
            out_ptr
            + pid_batch * STRIDE_OUT_BATCH
            + offs_head[:, None, None] * STRIDE_OUT_HEAD
            + offs_p[None, :, None] * STRIDE_OUT_HDIM
            + offs_n[None, None, :] * STRIDE_OUT_DSTATE
        )
        tl.store(
            out_ptrs,
            acc,
            mask=(
                head_ok[:, None, None]
                & p_ok[None, :, None]
                & n_ok[None, None, :]
            ),
        )


@triton.jit
def _chunk_state_varlen_nomask_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_seqlens_ptr,
    states_ptr,
    total_seqlen,
    hdim,
    dstate,
    nheads_ngroups_ratio,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_states_batch,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    NCHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    pid_c = tl.minimum(tl.maximum((end_idx - 1) // CHUNK_SIZE, 0), NCHUNKS - 1)
    seg_end = tl.minimum(
        tl.maximum(end_idx - pid_c * CHUNK_SIZE, 0), CHUNK_SIZE
    )
    seg_start = tl.minimum(
        tl.maximum(start_idx - pid_c * CHUNK_SIZE, 0), CHUNK_SIZE
    )

    b_base = b_ptr + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_base = x_ptr + pid_h * stride_x_head
    dt_base = dt_ptr + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    da_base = (
        dA_cumsum_ptr + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    )
    dA_cs_last = tl.load(
        da_base + tl.maximum(seg_end - 1, 0) * stride_dA_cs_csize
    ).to(tl.float32)

    # M / N 绕回，越界行列由 store 的掩码丢弃——不需要载入掩码。
    rows = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % hdim
    cols = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % dstate
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(CHUNK_SIZE, BLOCK_SIZE_K)):
        t = k_block * BLOCK_SIZE_K + offs_k
        valid = (t >= seg_start) & (t < seg_end)
        # 夹进 chunk 内，再夹进整个 packed 缓冲内（末块不满时后者才起作用）。
        t_in_chunk = tl.minimum(t, CHUNK_SIZE - 1)
        tok = tl.minimum(pid_c * CHUNK_SIZE + t_in_chunk, total_seqlen - 1)

        x = tl.load(
            x_base
            + rows[:, None] * stride_x_hdim
            + tok[None, :] * stride_x_seqlen
        )
        b = tl.load(
            b_base
            + tok[:, None] * stride_b_seqlen
            + cols[None, :] * stride_b_dstate
        ).to(tl.float32)
        dt_k = tl.load(dt_base + t_in_chunk * stride_dt_csize).to(tl.float32)
        dA_cs_k = tl.load(da_base + t_in_chunk * stride_dA_cs_csize).to(
            tl.float32
        )

        # 段外位置在这里置零——等价于掩码载入，但不走 mask-zero 路径。
        scale = tl.where(valid, tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0)
        b *= scale[:, None]
        acc += tl.dot(x.to(tl.float32), b, input_precision="ieee")

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptrs = (
        states_ptr
        + pid_b * stride_states_batch
        + pid_h * stride_states_head
        + offs_m[:, None] * stride_states_hdim
        + offs_n[None, :] * stride_states_dstate
    )
    tl.store(
        states_ptrs,
        states,
        mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
    )


def _block_for(extent, cap):
    for block in (128, 64, 32):
        if block <= cap and extent % block == 0:
            return block
    return 16


def _heads_per_tile(ratio, block_p):
    cap = 128 // block_p
    for heads in (8, 4, 2):
        if heads <= cap and heads <= ratio:
            return heads
    return 1


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1

    ratio = nheads // ngroups
    states = torch.empty(
        (batch, nheads, headdim, dstate),
        device=x.device,
        dtype=chunk_states.dtype,
    )
    sx, sb, sdt, sda, so = (
        x.stride(),
        B.stride(),
        dt.stride(),
        dA_cumsum.stride(),
        states.stride(),
    )
    block_p = max(16, min(128, triton.next_power_of_2(headdim)))
    heads_per_tile = _heads_per_tile(ratio, block_p)
    block_n = _block_for(dstate, _BLOCK_N_CAP)
    block_k = _block_for(chunk_size, _BLOCK_K_CAP)
    regular = (
        x.dtype != torch.float32
        and chunk_size >= 128
        and headdim >= 32
        and dstate >= 32
        and total_seqlen == nchunks * chunk_size
        and chunk_size % block_k == 0
        and headdim % block_p == 0
        and dstate % block_n == 0
        and ratio % heads_per_tile == 0
    )
    if regular:
        num_tasks = (
            batch
            * ngroups
            * triton.cdiv(ratio, heads_per_tile)
            * triton.cdiv(headdim, block_p)
            * triton.cdiv(dstate, block_n)
        )
        num_programs = min(num_tasks, _PERSISTENT_PROGRAMS)
        _chunk_state_varlen_cube_persistent_kernel[(num_programs,)](
            x,
            B,
            dt,
            dA_cumsum,
            cu_seqlens,
            states,
            sx[0],
            sx[1],
            sx[2],
            sb[0],
            sb[1],
            sb[2],
            sdt[0],
            sdt[1],
            sdt[2],
            sda[0],
            sda[1],
            sda[2],
            so[0],
            so[1],
            so[2],
            so[3],
            total_seqlen,
            num_tasks,
            nchunks,
            ngroups,
            ratio,
            chunk_size,
            headdim,
            dstate,
            heads_per_tile,
            block_p,
            block_n,
            block_k,
            num_warps=_PERSISTENT_WARPS,
            num_stages=1,
        )
    else:
        grid = (
            triton.cdiv(headdim, _BLOCK_M) * triton.cdiv(dstate, _BLOCK_N),
            batch,
            nheads,
        )
        _chunk_state_varlen_nomask_kernel[grid](
            x,
            B,
            dt,
            dA_cumsum,
            cu_seqlens,
            states,
            total_seqlen,
            headdim,
            dstate,
            ratio,
            sx[0],
            sx[1],
            sx[2],
            sb[0],
            sb[1],
            sb[2],
            sdt[1],
            sdt[0],
            sdt[2],
            sda[1],
            sda[0],
            sda[2],
            so[0],
            so[1],
            so[2],
            so[3],
            nchunks,
            chunk_size,
            _BLOCK_M,
            _BLOCK_N,
            _BLOCK_K,
            num_warps=4,
            num_stages=1,
        )
    return states


__all__ = ["chunk_state_varlen"]
