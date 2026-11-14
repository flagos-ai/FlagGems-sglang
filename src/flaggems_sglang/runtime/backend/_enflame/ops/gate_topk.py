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

import torch
import triton
import triton.language as tl

__all__ = ["gate_topk"]

# 燧原专用（v4，2026-11-12）。
# v3（2026-11-08）是一行一个 program 的 1D 选路（grid=(M,)，axis=0 归约）——
# 当年 v2 的 [BM,BN] 2D tile 被 11-07 20:22 工具链击穿 fresh compile
# （memref_ext.slice_pad_start 8 vs 7，P01a），被迫 1D 重写，见
# compiler_issues/sglang_gate_topk.md。P01a 已于 2026-11-10 定案解除
# （compiler_problems/P01a：外部 gcu-compiler-opt 对齐 8 段定义后重放全绿），
# v4 在 v3 基础上恢复 2D 行批臂，仅用于大 M：
#   - v3 的 grid=(M,) 在 m=512 被分发吞吐地板封顶（空 kernel 地板 25.6µs
#     @grid=512，实测 31.6/40.2µs）；2D [BLOCK_M, N] 锚定行批把 program 数
#     压缩 BLOCK_M 倍，m=512 k=4/8: 31.6→12.0 / 40.2→18.5µs，
#     m=4096: 231→29.0 / 307→48.6µs（do_bench median，bf16 N=256）。
#   - 两臂共用同一套 K 轮「tl.max + leftmost-argmax（tl.where + tl.min）」
#     选路（平手取最小列下标，与 torch.topk 语义一致），选中列置 -inf；
#     2D 臂输出攒进 [BLOCK_M, KP] 后整块 store；锚定尾块免行 mask，重叠行
#     幂等重算；BM*N>16384 回退 [32,N]（DSM 预算，与 topk_softmax 同界）。
#   - 分派（L600 实测）：M<=192 一律 1D（m=1/8/64=4.2~8.0µs 近发射地板，
#     m=128 2D 无收益）；M>192 且 N∈[16,512] 为 2 幂走 2D，其余 1D 兜底。
#   - 2D 臂首调对拍守卫：2D 写 scratch 与 1D（通用语义锚）逐位比对，
#     编译失败/错值则该配置在本进程内永回退 1D（防工具链级 2D 回归，
#     topk_softmax / moe_fused_gate 同款）。
#   - 行基址默认 int32（i32 安全域守卫，越界回退 int64，两臂同守卫）。
# 1D 臂细节（v3 注释归档）：load other=-inf 使死列天然不可选，N==BLOCK_N
# 整行免 mask（EVEN_N）；结果先攒 [KP] 向量再一次性 vector store（省 2K 次
# 标量 store）；num_warps=1：32 lane 内归约免跨 warp 树（nw2/4 实测慢
# 1.6~2 倍）。


@triton.jit
def _gate_topk_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    M,
    N,
    stride_xm,
    K: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    kc = tl.arange(0, KP)
    if WIDE:
        base = row.to(tl.int64) * stride_xm
        obase = row.to(tl.int64) * K
    else:
        base = row * stride_xm
        obase = row * K
    if EVEN_N:
        x = tl.load(x_ptr + base + cols).to(tl.float32)
    else:
        x = tl.load(
            x_ptr + base + cols, mask=cols < N, other=float("-inf")
        ).to(tl.float32)
    out_v = tl.zeros((KP,), dtype=tl.float32)
    out_i = tl.zeros((KP,), dtype=tl.int32)
    for i in tl.static_range(K):
        cur = tl.max(x, axis=0)
        if EVEN_N:
            cand = tl.where(x == cur, cols, BLOCK_N)
        else:
            cand = tl.where((x == cur) & (cols < N), cols, BLOCK_N)
        sel = tl.min(cand, axis=0)
        out_v = tl.where(kc == i, cur, out_v)
        out_i = tl.where(kc == i, sel, out_i)
        x = tl.where(cols == sel, float("-inf"), x)
    tl.store(
        val_ptr + obase + kc, out_v.to(val_ptr.dtype.element_ty), mask=kc < K
    )
    tl.store(idx_ptr + obase + kc, out_i, mask=kc < K)


@triton.jit
def _gate_topk_rows_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    # 锚定行批：最后一个 program 滑到 [M-BLOCK_M, M)，整块免行 mask；
    # 双覆盖行按相同确定性算法幂等重算。要求 M >= BLOCK_M（分派保证）。
    row0 = tl.minimum(tl.program_id(0) * BLOCK_M, M - BLOCK_M)
    rows = row0 + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    kc = tl.arange(0, KP)
    if WIDE:
        rbase = rows.to(tl.int64) * stride_xm
        obase = rows.to(tl.int64) * K
    else:
        rbase = rows * stride_xm
        obase = rows * K
    x = tl.load(x_ptr + rbase[:, None] + cols[None, :]).to(tl.float32)
    out_v = tl.zeros((BLOCK_M, KP), dtype=tl.float32)
    out_i = tl.zeros((BLOCK_M, KP), dtype=tl.int32)
    for i in tl.static_range(K):
        cur = tl.max(x, axis=1)
        cand = tl.where(x == cur[:, None], cols[None, :], BLOCK_N)
        sel = tl.min(cand, axis=1)
        out_v = tl.where(kc[None, :] == i, cur[:, None], out_v)
        out_i = tl.where(kc[None, :] == i, sel[:, None], out_i)
        x = tl.where(cols[None, :] == sel[:, None], float("-inf"), x)
    tl.store(
        val_ptr + obase[:, None] + kc[None, :],
        out_v.to(val_ptr.dtype.element_ty),
        mask=kc[None, :] < K,
    )
    tl.store(
        idx_ptr + obase[:, None] + kc[None, :], out_i, mask=kc[None, :] < K
    )


# ---------------------------------------------------------------------------
# Host dispatch
# ---------------------------------------------------------------------------

# 1D/2D 分派线与 2D BM 带（L600 实测，bf16 N=256，do_bench median）：
#   m=128 1D 10.3/12.8µs vs 2D 最优 10.5/16.2µs（1D 胜）；m=256 17.1/22.4 →
#   11.1/17.0µs（2D 胜）→ 分派线取 M>192（与 topk_softmax 同界）。
_ROWS_KERNEL_MIN_M = 192


def _launch_row_kernel(x, values, indices, M, N, k, wide):
    block_n = max(128, triton.next_power_of_2(N))
    _gate_topk_kernel[(M,)](
        x,
        values,
        indices,
        M,
        N,
        x.stride(0),
        K=k,
        KP=triton.next_power_of_2(k),
        BLOCK_N=block_n,
        WIDE=wide,
        EVEN_N=1 if N == block_n else 0,
        num_warps=1,
    )


def _rows_kernel_eligible(num_cols):
    # 2D 行批整块免 mask 装载 [BLOCK_M, N]，要求 N 为 2 幂；N>=16 避开
    # 窄向量病理区，N<=512 维持 BM*N<=16384 的 DSM 舒适预算（topk_softmax
    # 同界）。其余形状走 1D 兜底。
    return (
        16 <= num_cols <= 512
        and num_cols == triton.next_power_of_2(num_cols)
    )


# Per-process first-call validation state for the row-batch kernel:
# key -> True (validated, use directly) / False (fall back to the 1D kernel).
_ROWS_KERNEL_STATE = {}


def _launch_rows_kernel(x, values, indices, M, N, k, block_m, wide, num_warps):
    stride_xm = x.stride(0)
    kp = max(8, triton.next_power_of_2(k))
    key = (M, N, k, block_m, kp, wide, num_warps, x.dtype, stride_xm)
    grid = (triton.cdiv(M, block_m),)
    state = _ROWS_KERNEL_STATE.get(key)
    if state is True:
        _gate_topk_rows_kernel[grid](
            x,
            values,
            indices,
            M,
            stride_xm,
            K=k,
            KP=kp,
            BLOCK_M=block_m,
            BLOCK_N=N,
            WIDE=wide,
            num_warps=num_warps,
            num_stages=1,
        )
        return True
    if state is False:
        return False

    # First call for this configuration: compile+run the row-batch kernel
    # into scratch and cross-check it against the 1D kernel (generic-
    # semantics anchor) before adopting it.  A compile failure (e.g. an
    # environment-level 2-D-tile backend regression, the 2026-11-07 P01a
    # family) or a value mismatch permanently disables the row-batch path
    # for this configuration in this process.
    try:
        v_ref = torch.empty_like(values)
        i_ref = torch.empty_like(indices)
        _gate_topk_rows_kernel[grid](
            x,
            v_ref,
            i_ref,
            M,
            stride_xm,
            K=k,
            KP=kp,
            BLOCK_M=block_m,
            BLOCK_N=N,
            WIDE=wide,
            num_warps=num_warps,
            num_stages=1,
        )
    except Exception:
        _ROWS_KERNEL_STATE[key] = False
        return False
    _launch_row_kernel(x, values, indices, M, N, k, wide)
    ok = torch.equal(v_ref, values) and torch.equal(i_ref, indices)
    _ROWS_KERNEL_STATE[key] = bool(ok)
    # values/indices already hold the bitwise-equal 1D results on success.
    return ok


def gate_topk(x, k):
    M, N = x.shape
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    if M == 0:
        return values, indices
    wide = 1 if (M - 1) * x.stride(0) + N > 2**31 - 1 else 0

    if M > _ROWS_KERNEL_MIN_M and k <= 64 and _rows_kernel_eligible(N):
        # k<=64：超出走 1D（K 轮 static_range 全展开，2D 臂按 BM 倍放大 IR，
        # 大 k 的编译膨胀风险不值得；真实 gate top-k 远小于此）。
        # Measured on L600 (48 SP), N=256 bf16 (kernel-mode do_bench median,
        # 2026-11-12): m=512 [16,nw2] 12.0/18.5µs, [32,nw4] 11.8/18.6µs;
        # m=4096 [64,nw4] 29.0/48.6µs vs [32,nw4] 38.5/61.5µs.
        if M <= 512:
            block_m, rows_warps = 16, 2
        elif M <= 1536:
            block_m, rows_warps = 32, 4
        else:
            block_m, rows_warps = 64, 4
        if block_m * N > 16384:
            # fp32 tile 集（x/cand + 归约临时量）守住 ~448KB 双 program
            # DSM 预算（BLOCK_N=512 时 BM=64 会超界）。
            block_m, rows_warps = 32, 4
        if _launch_rows_kernel(x, values, indices, M, N, k, block_m, wide, rows_warps):
            return values, indices

    _launch_row_kernel(x, values, indices, M, N, k, wide)
    return values, indices
