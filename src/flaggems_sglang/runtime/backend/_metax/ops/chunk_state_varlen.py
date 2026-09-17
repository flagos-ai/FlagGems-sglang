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

_BLOCK_M_CAP = 128
_BLOCK_N_CAP = 128
_BLOCK_K_CAP = 32
_NUM_WARPS = 4
_NUM_STAGES = 3


@triton.jit
def _chunk_state_varlen_grouped_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    da_ptr,
    cu_seqlens_ptr,
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
    NCHUNKS: tl.constexpr,
    NHEADS: tl.constexpr,
    RATIO: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM: tl.constexpr,
    DSTATE: tl.constexpr,
    HEADS_PER_TILE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST_DOT: tl.constexpr,
    USE_BF16: tl.constexpr,
):
    block_m: tl.constexpr = HEADS_PER_TILE * BLOCK_P
    num_pid_n = tl.cdiv(DSTATE, BLOCK_N)
    num_pid_p = tl.cdiv(HEADDIM, BLOCK_P)

    pid_n = tl.program_id(0) % num_pid_n
    pid_hp = tl.program_id(0) // num_pid_n
    pid_p = pid_hp % num_pid_p
    pid_head_tile = pid_hp // num_pid_p
    pid_batch = tl.program_id(1)
    pid_group = tl.program_id(2)

    head_local = pid_head_tile * HEADS_PER_TILE + tl.arange(0, HEADS_PER_TILE)
    offs_head = pid_group * RATIO + head_local
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    head_ok = head_local < RATIO
    p_ok = offs_p < HEADDIM
    n_ok = offs_n < DSTATE

    end_idx = tl.load(cu_seqlens_ptr + pid_batch + 1)
    start_idx = tl.load(cu_seqlens_ptr + pid_batch)
    pid_chunk = tl.minimum(
        tl.maximum((end_idx - 1) // CHUNK_SIZE, 0),
        NCHUNKS - 1,
    )
    chunk_base = pid_chunk * CHUNK_SIZE
    seg_end = tl.minimum(tl.maximum(end_idx - chunk_base, 0), CHUNK_SIZE)
    seg_start = tl.minimum(
        tl.maximum(start_idx - chunk_base, 0),
        seg_end,
    )

    x_base = x_ptr + chunk_base * STRIDE_X_SEQLEN
    b_base = b_ptr + chunk_base * STRIDE_B_SEQLEN + pid_group * STRIDE_B_GROUP
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
    da_last_ptrs = (
        da_base
        + offs_head * STRIDE_DA_HEAD
        + tl.maximum(seg_end - 1, 0) * STRIDE_DA_CSIZE
    )
    da_last = tl.load(da_last_ptrs, mask=head_ok, other=0.0).to(tl.float32)

    acc = tl.zeros((block_m, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(CHUNK_SIZE, BLOCK_K)):
        offs_k_global = k_block * BLOCK_K + offs_k
        k_ok = (offs_k_global >= seg_start) & (offs_k_global < seg_end)
        x = tl.load(
            x_ptrs,
            mask=(
                head_ok[:, None, None]
                & p_ok[None, :, None]
                & k_ok[None, None, :]
            ),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptrs,
            mask=k_ok[:, None] & n_ok[None, :],
            other=0.0,
        ).to(tl.float32)
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
        )
        a = (x * scale[:, None, :]).reshape((block_m, BLOCK_K))
        if USE_BF16:
            acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        elif FAST_DOT:
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)
        else:
            acc = tl.dot(
                a.to(tl.float32),
                b.to(tl.float32),
                acc,
                input_precision="ieee",
            )

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
            head_ok[:, None, None] & p_ok[None, :, None] & n_ok[None, None, :]
        ),
    )


def _block_for(extent, cap):
    for block in (128, 64, 32):
        if block <= cap and extent % block == 0:
            return block
    return 16


def _heads_per_tile(ratio, block_p):
    cap = _BLOCK_M_CAP // block_p
    for heads in (8, 4, 2):
        if heads <= cap and heads <= ratio:
            return heads
    return 1


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    _, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups

    block_p = max(16, min(_BLOCK_M_CAP, triton.next_power_of_2(headdim)))
    heads_per_tile = _heads_per_tile(ratio, block_p)
    block_n = _block_for(dstate, _BLOCK_N_CAP)
    block_k = _block_for(chunk_size, _BLOCK_K_CAP)
    fast_dot = x.dtype == B.dtype
    use_bf16 = False

    out = torch.empty(
        (batch, nheads, headdim, dstate),
        device=x.device,
        dtype=chunk_states.dtype,
    )
    sx = x.stride()
    sb = B.stride()
    sdt = dt.stride()
    sda = dA_cumsum.stride()
    so = out.stride()
    grid = (
        triton.cdiv(ratio, heads_per_tile)
        * triton.cdiv(headdim, block_p)
        * triton.cdiv(dstate, block_n),
        batch,
        ngroups,
    )
    launch_options = {
        "num_warps": _NUM_WARPS,
        "num_stages": _NUM_STAGES,
        "pipeline": "cpasync",
    }
    _chunk_state_varlen_grouped_kernel[grid](
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
        nchunks,
        nheads,
        ratio,
        chunk_size,
        headdim,
        dstate,
        heads_per_tile,
        block_p,
        block_n,
        block_k,
        fast_dot,
        use_bf16,
        **launch_options,
    )
    return out


__all__ = ["chunk_state_varlen"]
