# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""chunk_state — Mamba2 SSM per-chunk hidden-state accumulation (dot main path).

问题+结论: 2026-11-07 20:22 工具链更新曾使 2D tile + ``tl.dot`` 主路 fresh
compile 全灭(``memref_ext.slice_transpose_start`` 8 vs 7,全形态无一幸存,
见 compiler_issues/sglang_chunk_state.md),11-08 以纯 1D 行累加核绕过——正确
但受 1D 结构地板限制(bench core 125.6/251.6 ms,vs torch 参考 1.94/3.79 ms)。
2026-11-09 起外部 gcu-compiler-opt 已与 libtriton 配套,tl.dot 恢复可用
(chunk_state_varlen 重写实证:fp16 dot 快路 bench 4.45×/5.32× 优于 generic),
故本实现回切 dot 主路,并移植 varlen 验证过的 GCU 安全形态:
段重定基仿射指针 + 纯上界 mask(无仿射下界,规避「压缩」地址左移误编译)、
单 K 迭代(BLOCK_K>=chunk,规避 K 循环 dot 累加 + 3D reshape 首批 head lane
错值)、fp32 乘后单次舍入进 fp16 dot(对齐 generic 精度方案)、
persistent 平铺 grid-stride 任务环、索引不用 tl.minimum 钳位(防 DMA 逐元素
~120× 劣化)。快路外的形状(小 chunk / fp32 / 非整 16)走 2D tile ieee fp32
dot 慢路。实测(bench core,GCU400 单卡,do_bench kernel 口径,48 programs=2
CTA/SIP——DSM 双 program 阈值实测解锁 ~1.5×):case0 (8,16,256,32,8,64,128)
bf16 0.3557ms、case1 (32,4,256,64,8,64,128) bf16 0.6976ms,vs torch 参考
5.46×/5.30×;vs 11-08 交付的 1D 绕过核(136.77/273.72ms 同口径复测)384×/392×;
vs 11-07 旧栈缓存快照(3.12/6.15ms,speedup 0.619)8.8×。pytest --quick 5/5
+ 奇形 stress 8/8(partial-N/ratio3/chunk192/fp32/K-loop);grid=24 对照臂
0.539/1.061ms(噪声带 ±0.1%,48 程序臂 -34% 为真信号)。

Semantics (identical to the PyTorch reference):

    states[b, c, h, p, n] = sum_t  x[b, c, t, h, p] * B[b, c, t, g(h), n]
                                   * exp(dA_last[b,h,c] - dA_cumsum[b,h,c,t])
                                   * dt[b,h,c,t]

Shapes:
    B         : [batch, seqlen, ngroups, dstate]
    x         : [batch, seqlen, nheads, headdim]
    dt        : [batch, nheads, nchunks, chunk_size]
    dA_cumsum : [batch, nheads, nchunks, chunk_size]
    states    : [batch, nchunks, nheads, headdim, dstate]  (output, float32)

Design (one kernel launch, no Torch compute on the path):
- Fast path (persistent kernel): flat grid-stride task loop over
  (batch, chunk, group, head-tile, p-tile, n-tile), single K iteration
  (BLOCK_K >= chunk_size <= 256), fp16 dot with fp32-accumulated scale.
  Heads of the same group fold into the M dimension (block_m =
  HEADS_PER_TILE * BLOCK_P) so the B tile is loaded once per group tile.
- Safe path (fallback kernel): plain 2D tiles + K loop + fp32 ieee dot
  for shapes outside the fast path's domain (fp32 inputs, small or
  oversized chunks).
- All tile addresses are affine from scalar bases; the only masks are pure
  upper bounds (head/p/n tails plus the chunk K bound) — the empirically
  reliable forms on this backend (see compiler_issues/sglang_chunk_state_varlen.md).
"""

import torch
import triton
import triton.language as tl

_PERSISTENT_PROGRAMS = 48
_PERSISTENT_WARPS = 1
_GRID_CAP = 65535


@triton.jit
def _chunk_state_persistent_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    da_ptr,
    out_ptr,
    # x strides [batch, seqlen, nheads, headdim]
    STRIDE_X_BATCH: tl.constexpr,
    STRIDE_X_SEQ: tl.constexpr,
    STRIDE_X_HEAD: tl.constexpr,
    STRIDE_X_HDIM: tl.constexpr,
    # B strides [batch, seqlen, ngroups, dstate]
    STRIDE_B_BATCH: tl.constexpr,
    STRIDE_B_SEQ: tl.constexpr,
    STRIDE_B_GROUP: tl.constexpr,
    STRIDE_B_DSTATE: tl.constexpr,
    # dt strides [batch, nheads, nchunks, chunk_size]
    STRIDE_DT_BATCH: tl.constexpr,
    STRIDE_DT_HEAD: tl.constexpr,
    STRIDE_DT_CHUNK: tl.constexpr,
    STRIDE_DT_TOKEN: tl.constexpr,
    # dA strides [batch, nheads, nchunks, chunk_size]
    STRIDE_DA_BATCH: tl.constexpr,
    STRIDE_DA_HEAD: tl.constexpr,
    STRIDE_DA_CHUNK: tl.constexpr,
    STRIDE_DA_TOKEN: tl.constexpr,
    # out strides [batch, nchunks, nheads, headdim, dstate]
    STRIDE_OUT_BATCH: tl.constexpr,
    STRIDE_OUT_CHUNK: tl.constexpr,
    STRIDE_OUT_HEAD: tl.constexpr,
    STRIDE_OUT_HDIM: tl.constexpr,
    STRIDE_OUT_DSTATE: tl.constexpr,
    # geometry
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
        pid_h = task_left % num_pid_h
        task_left = task_left // num_pid_h
        pid_group = task_left % NGROUPS
        task_left = task_left // NGROUPS
        pid_chunk = task_left % NCHUNKS
        pid_batch = task_left // NCHUNKS

        head_local = pid_h * HEADS_PER_TILE + offs_head_base
        offs_head = pid_group * RATIO + head_local
        offs_p = pid_p * BLOCK_P + offs_p_base
        offs_n = pid_n * BLOCK_N + offs_n_base
        head_ok = head_local < RATIO
        p_ok = offs_p < HEADDIM
        n_ok = offs_n < DSTATE
        k_ok = offs_k < CHUNK_SIZE

        # Scalar-rebased affine tiles: the chunk base (chunk start on the
        # seqlen axis) is folded into every pointer as a scalar, so tile
        # lanes stay affine with pure upper-bound masks only.
        chunk_base = pid_chunk * CHUNK_SIZE
        x_ptrs = (
            x_ptr
            + pid_batch * STRIDE_X_BATCH
            + offs_head[:, None, None] * STRIDE_X_HEAD
            + offs_p[None, :, None] * STRIDE_X_HDIM
            + (chunk_base + offs_k[None, None, :]) * STRIDE_X_SEQ
        )
        b_ptrs = (
            b_ptr
            + pid_batch * STRIDE_B_BATCH
            + pid_group * STRIDE_B_GROUP
            + offs_n[None, :] * STRIDE_B_DSTATE
            + (chunk_base + offs_k[:, None]) * STRIDE_B_SEQ
        )
        dt_ptrs = (
            dt_ptr
            + pid_batch * STRIDE_DT_BATCH
            + pid_chunk * STRIDE_DT_CHUNK
            + offs_head[:, None] * STRIDE_DT_HEAD
            + offs_k[None, :] * STRIDE_DT_TOKEN
        )
        da_ptrs = (
            da_ptr
            + pid_batch * STRIDE_DA_BATCH
            + pid_chunk * STRIDE_DA_CHUNK
            + offs_head[:, None] * STRIDE_DA_HEAD
            + offs_k[None, :] * STRIDE_DA_TOKEN
        )
        da_last = tl.load(
            da_ptr
            + pid_batch * STRIDE_DA_BATCH
            + pid_chunk * STRIDE_DA_CHUNK
            + offs_head * STRIDE_DA_HEAD
            + (CHUNK_SIZE - 1) * STRIDE_DA_TOKEN,
            mask=head_ok,
            other=0.0,
        ).to(tl.float32)

        x = tl.load(
            x_ptrs,
            mask=(
                head_ok[:, None, None]
                & p_ok[None, :, None]
                & k_ok[None, None, :]
            ),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_ok[:, None] & n_ok[None, :],
            other=0.0,
        )
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
            k_ok[None, :],
            tl.exp(da_last[:, None] - da) * dt,
            0.0,
        )
        a = (x * scale[:, None, :]).to(tl.float16).reshape((block_m, BLOCK_K))
        acc = tl.dot(a, b.to(tl.float16))

        acc = acc.reshape((HEADS_PER_TILE, BLOCK_P, BLOCK_N))
        out_ptrs = (
            out_ptr
            + pid_batch * STRIDE_OUT_BATCH
            + pid_chunk * STRIDE_OUT_CHUNK
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
def _chunk_state_fallback_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_ptr,
    states_ptr,
    # sizes
    headdim,
    dstate,
    nheads,
    nchunks,
    chunk_size,
    ratio,
    total_work,
    num_mn,
    num_pid_n,
    # x strides [batch, seqlen, nheads, headdim]
    stride_xb,
    stride_xt,
    stride_xh,
    stride_xp,
    # B strides [batch, seqlen, ngroups, dstate]
    stride_bb,
    stride_bt,
    stride_bg,
    stride_bn,
    # dt / dA strides [batch, nheads, nchunks, chunk_size]
    stride_dtb,
    stride_dth,
    stride_dtc,
    stride_dtt,
    stride_dab,
    stride_dah,
    stride_dac,
    stride_dat,
    # states strides [batch, nchunks, nheads, headdim, dstate]
    stride_sb,
    stride_sc,
    stride_sh,
    stride_sp,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    num_programs = tl.num_programs(0)
    for work in tl.range(tl.program_id(0), total_work, num_programs):
        mn = work % num_mn
        pid_m = mn // num_pid_n
        pid_n = mn % num_pid_n
        bch = work // num_mn
        h = bch % nheads
        tmp = bch // nheads
        c = tmp % nchunks
        b = tmp // nchunks
        grp = h // ratio

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_base = x_ptr + b * stride_xb + (c * chunk_size) * stride_xt \
            + h * stride_xh
        b_base = b_ptr + b * stride_bb + (c * chunk_size) * stride_bt \
            + grp * stride_bg
        dt_base = dt_ptr + b * stride_dtb + h * stride_dth + c * stride_dtc
        da_base = dA_ptr + b * stride_dab + h * stride_dah + c * stride_dac
        o_base = states_ptr + b * stride_sb + c * stride_sc + h * stride_sh

        dA_last = tl.load(da_base + (chunk_size - 1) * stride_dat).to(
            tl.float32
        )

        x_ptrs = x_base + offs_m[:, None] * stride_xp \
            + offs_k[None, :] * stride_xt
        b_ptrs = b_base + offs_k[:, None] * stride_bt \
            + offs_n[None, :] * stride_bn
        dt_ptrs = dt_base + offs_k * stride_dtt
        da_ptrs = da_base + offs_k * stride_dat

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, chunk_size, BLOCK_K):
            k_ok = offs_k < chunk_size - k
            x = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < headdim) & k_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            bmat = tl.load(
                b_ptrs,
                mask=k_ok[:, None] & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dtv = tl.load(dt_ptrs, mask=k_ok, other=0.0).to(tl.float32)
            dav = tl.load(da_ptrs, mask=k_ok, other=0.0).to(tl.float32)
            scale = tl.where(k_ok, tl.exp(dA_last - dav) * dtv, 0.0)
            acc += tl.dot(
                x * scale[None, :], bmat, input_precision="ieee"
            )
            x_ptrs += BLOCK_K * stride_xt
            b_ptrs += BLOCK_K * stride_bt
            dt_ptrs += BLOCK_K * stride_dtt
            da_ptrs += BLOCK_K * stride_dat

        tl.store(
            o_base + offs_m[:, None] * stride_sp + offs_n[None, :] * stride_sn,
            acc,
            mask=(offs_m[:, None] < headdim) & (offs_n[None, :] < dstate),
        )


def _heads_per_tile(ratio, block_p):
    cap = 128 // block_p
    for heads in (8, 4, 2):
        if heads <= cap and heads <= ratio:
            return heads
    return 1


def chunk_state(B, x, dt, dA_cumsum):
    """Mamba2 chunk-state: accumulate x * (decay*dt-scaled B) over each chunk.

    Args:
        B:         [batch, seqlen, ngroups, dstate] SSM state-projection matrix.
        x:         [batch, seqlen, nheads, headdim].
        dt:        [batch, nheads, nchunks, chunk_size].
        dA_cumsum: [batch, nheads, nchunks, chunk_size].

    Returns:
        [batch, nchunks, nheads, headdim, dstate] float32 tensor.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups
    if seqlen != nchunks * chunk_size:
        raise ValueError(
            f"seqlen {seqlen} != nchunks {nchunks} * chunk_size {chunk_size}"
        )
    states = torch.empty(
        (batch, nchunks, nheads, headdim, dstate),
        device=x.device,
        dtype=torch.float32,
    )

    regular = (
        x.dtype != torch.float32
        and 128 <= chunk_size <= 256
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
        block_k = 128 if chunk_size <= 128 else 256
        num_tasks = (
            batch
            * nchunks
            * ngroups
            * triton.cdiv(ratio, heads_per_tile)
            * triton.cdiv(headdim, block_p)
            * triton.cdiv(dstate, block_n)
        )
        num_programs = min(num_tasks, _PERSISTENT_PROGRAMS)
        _chunk_state_persistent_kernel[(num_programs,)](
            x,
            B,
            dt,
            dA_cumsum,
            states,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(1),
            dA_cumsum.stride(2),
            dA_cumsum.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
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
        block_m = max(16, min(64, triton.next_power_of_2(headdim)))
        block_n = max(16, min(64, triton.next_power_of_2(dstate)))
        block_k = max(16, min(32, triton.next_power_of_2(chunk_size)))
        num_pid_m = triton.cdiv(headdim, block_m)
        num_pid_n = triton.cdiv(dstate, block_n)
        num_mn = num_pid_m * num_pid_n
        total_work = batch * nchunks * nheads * num_mn
        grid = (min(total_work, _GRID_CAP),)
        _chunk_state_fallback_kernel[grid](
            x,
            B,
            dt,
            dA_cumsum,
            states,
            headdim,
            dstate,
            nheads,
            nchunks,
            chunk_size,
            ratio,
            total_work,
            num_mn,
            num_pid_n,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(1),
            dA_cumsum.stride(2),
            dA_cumsum.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
    return states


__all__ = ["chunk_state"]
