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

_PERSISTENT_PROGRAMS = 24
_PERSISTENT_WARPS = 1
_BLOCK_K_CAP = 128


@triton.jit
def _chunk_state_varlen_persistent_kernel(
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
    offs_head_base = tl.arange(0, HEADS_PER_TILE)
    offs_p_base = tl.arange(0, BLOCK_P)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

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

        head_local = pid_head_tile * HEADS_PER_TILE + offs_head_base
        offs_head = pid_group * RATIO + head_local
        offs_p = pid_p * BLOCK_P + offs_p_base
        offs_n = pid_n * BLOCK_N + offs_n_base
        head_ok = head_local < RATIO
        p_ok = offs_p < HEADDIM
        n_ok = offs_n < DSTATE

        end_idx = tl.load(cu_ptr + pid_batch + 1)
        start_idx = tl.load(cu_ptr + pid_batch)
        pid_chunk = tl.minimum(
            tl.maximum((end_idx - 1) // CHUNK_SIZE, 0),
            NCHUNKS - 1,
        )
        chunk_base = pid_chunk * CHUNK_SIZE
        seg_end = tl.minimum(tl.maximum(end_idx - chunk_base, 0), CHUNK_SIZE)
        seg_start = tl.minimum(tl.maximum(start_idx - chunk_base, 0), seg_end)

        x_base = x_ptr + chunk_base * STRIDE_X_SEQLEN
        b_base = (
            b_ptr + chunk_base * STRIDE_B_SEQLEN + pid_group * STRIDE_B_GROUP
        )
        dt_base = dt_ptr + pid_chunk * STRIDE_DT_CHUNK
        da_base = da_ptr + pid_chunk * STRIDE_DA_CHUNK
        x_ptrs = (
            x_base
            + offs_head[:, None, None] * STRIDE_X_HEAD
            + offs_p[None, :, None] * STRIDE_X_HDIM
            + offs_k[None, None, :] * STRIDE_X_SEQLEN
        )
        b_ptrs = (
            b_base
            + offs_k[:, None] * STRIDE_B_SEQLEN
            + offs_n[None, :] * STRIDE_B_DSTATE
        )
        dt_ptrs = (
            dt_base
            + offs_head[:, None] * STRIDE_DT_HEAD
            + offs_k[None, :] * STRIDE_DT_CSIZE
        )
        da_ptrs = (
            da_base
            + offs_head[:, None] * STRIDE_DA_HEAD
            + offs_k[None, :] * STRIDE_DA_CSIZE
        )
        da_last = tl.load(
            da_base
            + offs_head * STRIDE_DA_HEAD
            + tl.maximum(seg_end - 1, 0) * STRIDE_DA_CSIZE,
            mask=head_ok,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((block_m, BLOCK_N), dtype=tl.float32)
        for k_block in tl.range(0, tl.cdiv(CHUNK_SIZE, BLOCK_K)):
            k_global = k_block * BLOCK_K + offs_k
            k_ok = (k_global >= seg_start) & (k_global < seg_end)
            x = tl.load(
                x_ptrs,
                mask=(
                    head_ok[:, None, None]
                    & p_ok[None, :, None]
                    & k_ok[None, None, :]
                ),
                other=0.0,
            ).to(tl.float16)
            b = tl.load(
                b_ptrs,
                mask=k_ok[:, None] & n_ok[None, :],
                other=0.0,
            ).to(tl.float16)
            dt = tl.load(
                dt_ptrs,
                mask=head_ok[:, None] & k_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            da = tl.load(
                da_ptrs,
                mask=head_ok[:, None] & k_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            scale = tl.where(
                head_ok[:, None] & k_ok[None, :],
                tl.exp(da_last[:, None] - da) * dt,
                0.0,
            ).to(tl.float16)
            a = (x * scale[:, None, :]).reshape((block_m, BLOCK_K))
            acc = tl.dot(a, b, acc)

            x_ptrs += BLOCK_K * STRIDE_X_SEQLEN
            b_ptrs += BLOCK_K * STRIDE_B_SEQLEN
            dt_ptrs += BLOCK_K * STRIDE_DT_CSIZE
            da_ptrs += BLOCK_K * STRIDE_DA_CSIZE

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
def _chunk_state_varlen_fallback_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    da_ptr,
    cu_ptr,
    out_ptr,
    hdim,
    dstate,
    chunk_size,
    ratio,
    sx_t,
    sx_h,
    sx_p,
    sb_t,
    sb_g,
    sb_n,
    sdt_c,
    sdt_h,
    sdt_t,
    sda_c,
    sda_h,
    sda_t,
    so_b,
    so_h,
    so_p,
    so_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    num_pid_n = tl.cdiv(dstate, BLOCK_N)
    pid_m = tl.program_id(0) // num_pid_n
    pid_n = tl.program_id(0) % num_pid_n
    end_idx = tl.load(cu_ptr + pid_b + 1)
    start_idx = tl.load(cu_ptr + pid_b)
    pid_c = (end_idx - 1) // chunk_size
    chunk_base = pid_c * chunk_size
    seg_end = end_idx - chunk_base
    seg_start = tl.maximum(start_idx - chunk_base, 0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = (
        x_ptr
        + chunk_base * sx_t
        + pid_h * sx_h
        + offs_m[:, None] * sx_p
        + offs_k[None, :] * sx_t
    )
    b_ptrs = (
        b_ptr
        + chunk_base * sb_t
        + (pid_h // ratio) * sb_g
        + offs_k[:, None] * sb_t
        + offs_n[None, :] * sb_n
    )
    dt_ptrs = dt_ptr + pid_c * sdt_c + pid_h * sdt_h + offs_k * sdt_t
    da_base = da_ptr + pid_c * sda_c + pid_h * sda_h
    da_ptrs = da_base + offs_k * sda_t
    da_last = tl.load(da_base + tl.maximum(seg_end - 1, 0) * sda_t).to(
        tl.float32
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, seg_end, BLOCK_K):
        k_ok = (offs_k < seg_end - k) & (offs_k >= seg_start - k)
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & k_ok[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_ok[:, None] & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(dt_ptrs, mask=k_ok, other=0.0).to(tl.float32)
        da = tl.load(da_ptrs, mask=k_ok, other=0.0).to(tl.float32)
        scale = tl.where(k_ok, tl.exp(da_last - da) * dt, 0.0)
        acc += tl.dot(
            x.to(tl.float32), b * scale[:, None], input_precision="ieee"
        )
        x_ptrs += BLOCK_K * sx_t
        b_ptrs += BLOCK_K * sb_t
        dt_ptrs += BLOCK_K * sdt_t
        da_ptrs += BLOCK_K * sda_t
    out_ptrs = (
        out_ptr
        + pid_b * so_b
        + pid_h * so_h
        + offs_m[:, None] * so_p
        + offs_n[None, :] * so_n
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
    )


def _heads_per_tile(ratio, block_p):
    cap = 128 // block_p
    for heads in (8, 4, 2):
        if heads <= cap and heads <= ratio:
            return heads
    return 1


def _block_for(extent, cap):
    for block in (128, 64, 32):
        if block <= cap and extent % block == 0:
            return block
    return 16


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    _, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups
    out = torch.empty(
        (batch, nheads, headdim, dstate),
        device=x.device,
        dtype=chunk_states.dtype,
    )
    sx, sb, sdt, sda, so = (
        x.stride(),
        B.stride(),
        dt.stride(),
        dA_cumsum.stride(),
        out.stride(),
    )

    regular = (
        x.dtype != torch.float32
        and chunk_size >= 128
        and headdim >= 32
        and dstate >= 32
        and chunk_size % 16 == 0
        and headdim % 16 == 0
        and dstate % 16 == 0
    )
    if regular:
        block_p = max(16, min(128, triton.next_power_of_2(headdim)))
        heads_per_tile = _heads_per_tile(ratio, block_p)
        block_n = 128 if dstate % 128 == 0 else 64
        block_k = _block_for(chunk_size, _BLOCK_K_CAP)
        num_tasks = (
            batch
            * ngroups
            * triton.cdiv(ratio, heads_per_tile)
            * triton.cdiv(headdim, block_p)
            * triton.cdiv(dstate, block_n)
        )
        num_programs = min(num_tasks, _PERSISTENT_PROGRAMS)
        _chunk_state_varlen_persistent_kernel[(num_programs,)](
            x,
            B,
            dt,
            dA_cumsum,
            cu_seqlens,
            out,
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
            triton.cdiv(headdim, 64) * triton.cdiv(dstate, 64),
            batch,
            nheads,
        )
        _chunk_state_varlen_fallback_kernel[grid](
            x,
            B,
            dt,
            dA_cumsum,
            cu_seqlens,
            out,
            headdim,
            dstate,
            chunk_size,
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
            64,
            64,
            32,
            num_warps=4,
            num_stages=1,
        )
    return out


__all__ = ["chunk_state_varlen"]
