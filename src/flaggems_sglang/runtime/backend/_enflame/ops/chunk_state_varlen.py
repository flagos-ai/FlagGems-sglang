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

"""chunk_state_varlen — Mamba2 varlen per-sequence chunk-state accumulation.

问题+结论(2026-11-09):seed 版双 kernel 在 GCU 上 5/5 全 FAIL(基线 0/5,
max|Δ|达 1736),根因是两个 enflame 后端静默误编译,均已被探针实证并绕过:
①2D/3D tile load 的 mask 含**仿射下界**(`offs_k >= seg_start - k`)时,
lowering 把下界折叠成地址左移(「压缩」语义:b=1 段 [3,8) 的 x/b tile 整体
左移 3 lane;seg_start=0 的段位移为零故恰好全对;1D load 同形 mask 正确;
mask lane 实测不读内存、other=0 生效)——与 compiler_issues/
sglang_causal_conv1d_fn.md 同族;②persistent 核 `tl.range` K 循环 ≥2 迭代
携带 dot 累加 + 3D load/reshape 形态下,首批 head lane 出现确定性错值
(~7% 元素,垃圾值至 241;单迭代免疫)。修法:tile load 改**段重定基仿射
指针**(基址含 chunk_base+seg_start,K 相对化)+ **纯上界 mask**(k 维
`offs_k < seg_len`,无下界、无循环变量减法——规避压缩降级;m/n 维上界
mask 保留),K 循环压成单迭代(BLOCK_K≥chunk_size,chunk≤256 走快路,
超出走 fallback);fp16 dot 改 fp32 乘后单次舍入(x*scale 在 fp32 算完再
.to(fp16),对齐 generic 精度方案)。注意:索引不得用 tl.minimum 钳位
(非仿射向量触发 DMA 逐元素降级,实测 ~120× 劣化),段外 lane 一律靠
上界 mask 的 other=0 + scale where 归零。验证:pytest --quick 5/5 绿
(基线 0/5);17 组 stress 全过(快/慢两路×奇形×ratio3/1×fp32/fp16×
chunk512×快路部分段);双路 enflame 5/5 vs generic 2/5(generic 的 3 挂
为同一后端缺陷命中)=完全修复;bench core 0.0464/0.2886ms(基线
0.0636/0.4263ms,-27%/-32%;vs generic 0.2065/1.5343ms,4.45×/5.32× 快)。
编译器档案:compiler_issues/sglang_chunk_state_varlen.md。

问题+结论(2026-11-13 E-1 二次优化):首轮后快路仍以 24 programs 跑
DSM staging 128KB(≤448KB)的 tile——姊妹 op chunk_state 2026-11-10 已
实证同构 tile 可解锁双 program/SIP(48-way)。本轮回移植:`_PERSISTENT_
PROGRAMS` 24→48,数值逐任务位级不变(grid-stride 步长仅重排任务归属)。
bench core 0.0467/0.2863→0.0339/0.1895ms(-27%/-34%,两 fresh 复跑
噪声带 ±1%/±0.2%,效应 >20× 带宽;vs generic 0.2068/1.5446ms=6.1×/
8.2× 快);pytest --quick 5/5 不回退。否决项(实测即回滚):num_warps=2
(+107%/+162%)、num_stages=2(+2.5%/+0.9%)、FULL_TILES 常量去 mask
(+2.1%/+0.4%,谓词税在此 DMA 形态不成立)。generic 精度腿当日重跑为
ABORT(11-09 时 PARTIAL 2/5——同一仿射下界误编译在当前工具链下由错值
升级为 device abort,bench 形状 seg_start=0 恰好免疫故 bench 腿正常),
分类不变:generic 非 PASS → enflame PASS = 完全修复。

Semantics (identical to the PyTorch reference):

    states[b, h, p, n] = sum_t  x[start_b+t, h, p] * B[start_b+t, g(h), n]
                                  * exp(dA_last - dA_cumsum[h, c_b, t_rel])
                                  * dt[h, c_b, t_rel]

    where [start_b, end_b) = cu_seqlens[b:b+2], c_b = (end_b-1)//chunk_size
    (the chunk holding the sequence's last token; sequences never straddle
    chunk boundaries per this problem's documented scope), t_rel indexes
    dt/dA_cumsum chunk-relative while x/B use global token positions.

Design (segment-rebased affine tiles, one kernel launch, no Torch compute):
- Fast path (persistent kernel): flat grid-stride task loop over
  (batch, group, head-tile, p-tile, n-tile), single K iteration
  (BLOCK_K >= chunk_size <= 256), fp16 dot with fp32-accumulated scale.
- Safe path (fallback kernel): plain 2D tiles + fp32 ieee dot for shapes
  outside the fast path's domain (fp32 inputs, small/oversized chunks).
- All tile addresses are affine from a scalar segment base; the only masks
  are pure upper bounds (k: segment length; head/p/n: tail tiles) plus the
  1D segment `tl.where` on the scale — the empirically reliable forms on
  this backend.
"""

import torch
import triton
import triton.language as tl

_PERSISTENT_PROGRAMS = 48
_PERSISTENT_WARPS = 1


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
        seg_len = seg_end - seg_start
        seg_last = tl.maximum(seg_end - 1, 0)

        # Segment-rebased affine tiles: lane j of the K axis reads chunk-
        # relative position seg_start + j; the pure upper-bound k mask
        # (j < seg_len) keeps every unmasked lane inside [seg_start,
        # seg_end) without any affine lower bound (which this backend
        # miscompiles into an address shift).
        x_ptrs = (
            x_ptr
            + (chunk_base + seg_start) * STRIDE_X_SEQLEN
            + offs_head[:, None, None] * STRIDE_X_HEAD
            + offs_p[None, :, None] * STRIDE_X_HDIM
            + offs_k[None, None, :] * STRIDE_X_SEQLEN
        )
        b_ptrs = (
            b_ptr
            + (chunk_base + seg_start) * STRIDE_B_SEQLEN
            + pid_group * STRIDE_B_GROUP
            + offs_k[:, None] * STRIDE_B_SEQLEN
            + offs_n[None, :] * STRIDE_B_DSTATE
        )
        dt_ptrs = (
            dt_ptr
            + pid_chunk * STRIDE_DT_CHUNK
            + seg_start * STRIDE_DT_CSIZE
            + offs_head[:, None] * STRIDE_DT_HEAD
            + offs_k[None, :] * STRIDE_DT_CSIZE
        )
        da_ptrs = (
            da_ptr
            + pid_chunk * STRIDE_DA_CHUNK
            + seg_start * STRIDE_DA_CSIZE
            + offs_head[:, None] * STRIDE_DA_HEAD
            + offs_k[None, :] * STRIDE_DA_CSIZE
        )
        da_last = tl.load(
            da_ptr
            + pid_chunk * STRIDE_DA_CHUNK
            + offs_head * STRIDE_DA_HEAD
            + seg_last * STRIDE_DA_CSIZE,
            mask=head_ok,
            other=0.0,
        ).to(tl.float32)

        k_ok = offs_k < seg_len
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
    nchunks,
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
    pid_c = tl.minimum(tl.maximum((end_idx - 1) // chunk_size, 0), nchunks - 1)
    chunk_base = pid_c * chunk_size
    seg_end = tl.minimum(tl.maximum(end_idx - chunk_base, 0), chunk_size)
    seg_start = tl.minimum(tl.maximum(start_idx - chunk_base, 0), seg_end)
    seg_len = seg_end - seg_start
    seg_last = tl.maximum(seg_end - 1, 0)
    seg_base = chunk_base + seg_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = (
        x_ptr
        + seg_base * sx_t
        + pid_h * sx_h
        + offs_m[:, None] * sx_p
        + offs_k[None, :] * sx_t
    )
    b_ptrs = (
        b_ptr
        + seg_base * sb_t
        + (pid_h // ratio) * sb_g
        + offs_k[:, None] * sb_t
        + offs_n[None, :] * sb_n
    )
    dt_ptrs = (
        dt_ptr
        + pid_c * sdt_c
        + pid_h * sdt_h
        + seg_start * sdt_t
        + offs_k * sdt_t
    )
    da_base = da_ptr + pid_c * sda_c + pid_h * sda_h + seg_start * sda_t
    da_ptrs = da_base + offs_k * sda_t
    da_last = tl.load(
        da_ptr + pid_c * sda_c + pid_h * sda_h + seg_last * sda_t
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, seg_len, BLOCK_K):
        k_ok = offs_k < seg_len - k
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
            nchunks,
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
