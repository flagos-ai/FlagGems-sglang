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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""tiny_n_gemm: thin-M bf16 GEMM for decode-phase projections.

GCU rewrite (2026-11-08, gcu410 / triton 3.6 + flagtree). The stock
implementation (one program per [16, 16] output tile looping tl.dot over
K in BLOCK_K=512 steps, plus a 1-D mul+sum GEMV for m == 1) runs at
~290-460 GB/s effective on the bench shapes; the m == 1 GEMV at ~160-270
GB/s. Root causes measured on this stack:

  * a runtime K loop around tl.dot serialises load->MMA rounds (the SIP
    call barrier blocks software pipelining), and the [16, 16] tile has
    strong N-diseconomies (N=32 dots are ~1.35x faster than N=16 despite
    2x the MACs);
  * the 1-D GEMV pays the vector-domain tax (fp32 upcast + tl.sum tree
    per 4-row slab) that the fused_moe_gemm case documented at ~5x tile
    traffic.

This rewrite dispatches two fast paths on top of the untouched generic
fallback:

  * ATen native GEMM for bf16 with N <= 512: topsprof shows a single
    kernel (op::matmul400_l1_split_k_kernel, small-M specialised, the
    w.t() transpose handled in-kernel - no copy). 14.5-15.3 us across
    all bench m at n=512/k=7168 (vs 25.2-46.7 us stock). Routed only in
    this N band: the native m<=2 tiers cliff at wider N (47.4 us at
    n=1536 vs 37.3 us for the Triton path below; 33.5 us already at
    n=640), while the Triton path is smooth in N.
  * A Triton full-K single-dot MMA kernel (the verified fused_moe_gemm
    form) for every other float shape with K <= 8192: the [16, BLOCK_K]
    activation tile is staged once per program (block_ptr, rows >= M
    zero-padded - runtime row offsets on the M axis miscompile, so the
    row base goes into the base pointer) and dotted against a transposed
    [BLOCK_K, BLOCK_N] block_ptr view of W; BLOCK_N=32, BLOCK_K =
    next_pow2(K), fp32 accumulator, any out dtype (this is what serves
    the fp32-out test case: native bf16 mm output is bf16-rounded and
    cannot meet the 1e-4 fp32 tolerance).

    Multi-dot-per-program variants are NOT usable on this toolchain:
    K-chunked accumulators (static_range acc chains AND straight-line
    dot+dot+add reorders) all produced data-dependent NaN in the last
    output column (compiler_issues/sglang_w8a8_fp32_dot_loop.md family),
    while the single-full-K-dot form passed every bit-level probe
    (8 seeds x m in {1..33} x n in {512, 1536}).

Measured (official bench protocol, kernel mode, warmup 1000ms/iter
100ms, bf16, k=7168, gcu410):

  m x n      stock      this rewrite
  1  x 512   46.7 us    14.5 us   (3.2x)
  1  x 1536  82.7 us    37.3 us   (2.2x)
  2+ x 512   25.2 us    14.6 us   (1.7x)
  2+ x 1536  50.3 us    37.4 us   (1.3x)

The generic kernels below are kept verbatim as the K > 8192 fallback
(they still compile on this stack once the dialect shim below is
installed).
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang import runtime as _runtime

# ===========================================================================
# GCU compiler-toolchain dialect compatibility shim
# ===========================================================================
# The installed FlagTree wheel emits MLIR where several `memref_ext.*` /
# `gcu.matmul` ops carry a trailing zero operand segment, while the external
# `gcu-compiler-opt` tool shipped in /opt/triton_gcu still registers the
# older operand layouts. The mismatch makes every kernel with 2-D tensor
# accesses (block pointers, masked 2-D loads, tl.dot tiles) fail to compile
# with "operandSegmentSizes ... size mismatch". The shim rewrites the
# generated MLIR text to the layout the external tool accepts (ported
# verbatim from the vllm enflame backend, router_gemm commit d1a7005, where
# it is production-verified; see compiler_issues/sglang_fused_moe_gemm.md).
# It is idempotent and leaves already-compatible IR (e.g. pure 1-D kernels)
# untouched.

import re

_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if op_name in ("memref.reinterpret_cast", "memref.subview") and not seg_match:
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = len([x for x in opers.group(1).split(",") if x.strip()]) if opers else 1
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(_fix_gcu_dialect_line(l) for l in content.split("\n"))
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-11 重放剥离: 私有 GCUIR shim 剥尾零
# 误伤已修复工具链（同 fused_moe_gemm 重放实证）；现行策略 = 禁用全部 op 私有
# shim（5dc28b9），剥离后 2D/block_ptr/dot 原生可编译。注释保留现场。


@triton.jit
def _tiny_n_mma_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program = one [BLOCK_M, BLOCK_N] output tile over the FULL K
    range in a single tl.dot (verified form; see module docstring)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    row0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    p_a = tl.make_block_ptr(
        x_ptr + row0.to(tl.int64) * stride_xm,
        (M - row0, K),
        (stride_xm, 1),
        (0, 0),
        (BLOCK_M, BLOCK_K),
        (1, 0),
    )
    a = tl.load(p_a, boundary_check=(0, 1))
    p_b = tl.make_block_ptr(
        w_ptr,
        (K, N),
        (1, stride_wn),
        (0, n0),
        (BLOCK_K, BLOCK_N),
        (0, 1),
    )
    b = tl.load(p_b, boundary_check=(0, 1))
    acc = tl.dot(a, b, out_dtype=tl.float32)

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ATen dispatch band: the native small-M GEMM tier is fastest up to
# N = 512 (measured cliff at n = 640: 33.5 us vs 21 us Triton).
_MM_MAX_N = 512

# Triton MMA path: BLOCK_N = 32 keeps the per-program DSM at
# 2 * BLOCK_K * (16 + 32) B (786432 B at BLOCK_K = 8192, inside the
# 917504 B limit); BLOCK_K covers all of K in one dot.
_MMA_BLOCK_N = 32
_MMA_MAX_BK = 8192


@triton.jit
def _tiny_n_dot_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    IEEE: tl.constexpr,
):
    """One program computes a [BLOCK_M, BLOCK_N] output tile over."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_col = offs_n[:, None] < N
    n_row = offs_n[None, :] < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    # w is [N, K] and K is the contiguous axis, so load the [BLOCK_N, BLOCK_K]
    # tile in its natural layout and transpose it in registers.
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_N:
        if EVEN_K:
            for _ in range(0, K, BLOCK_K):
                a = tl.load(x_ptrs, mask=m_mask, other=0.0)
                b = tl.load(w_ptrs)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
        else:
            for k0 in range(0, K, BLOCK_K):
                k_mask = (offs_k + k0) < K
                a = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
                b = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
    else:
        if EVEN_K:
            for _ in range(0, K, BLOCK_K):
                a = tl.load(x_ptrs, mask=m_mask, other=0.0)
                b = tl.load(w_ptrs, mask=n_col, other=0.0)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
        else:
            for k0 in range(0, K, BLOCK_K):
                k_mask = (offs_k + k0) < K
                a = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
                b = tl.load(w_ptrs, mask=n_col & k_mask[None, :], other=0.0)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, out, mask=m_mask & n_row)


@triton.jit
def _tiny_n_dot_splitk_kernel(
    x_ptr,
    w_ptr,
    part_ptr,  # [split_k, M, N] fp32 partial sums
    M,
    N,
    K,
    split_k,
    k_steps,  # uniform number of BLOCK_K steps per K-chunk
    stride_xm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    """One program computes a [BLOCK_M, BLOCK_N] tile over one K-chunk."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tile = num_pid_m * num_pid_n

    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // tile
    pid_k = (pid_k + pid_m * num_pid_n + pid_n) % split_k

    k_start = pid_k * k_steps * BLOCK_K
    k_end = tl.minimum(k_start + k_steps * BLOCK_K, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_col = offs_n[:, None] < N
    n_row = offs_n[None, :] < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_start + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_start + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(k_start, k_end, BLOCK_K):
        a = tl.load(x_ptrs, mask=m_mask, other=0.0)
        b = tl.load(w_ptrs, mask=n_col, other=0.0)
        if IEEE:
            acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
        else:
            acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    part_ptrs = (
        part_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    )
    tl.store(part_ptrs, acc, mask=m_mask & n_row)


@triton.jit
def _tiny_n_reduce_kernel(
    part_ptr,  # [split_k, M, N] fp32
    out_ptr,
    total,
    split_k,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, split_k):
        acc += tl.load(part_ptr + s * total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tiny_n_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """m == 1 fast path: broadcast-MAC over a [BLOCK_N, BLOCK_K] fp32."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    n_mask = offs_n[:, None] < N
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    if EVEN_K:
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(x_ptr + k0 + offs_k).to(tl.float32)
            b = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            acc += b * a[None, :]
            w_ptrs += BLOCK_K
    else:
        for k0 in range(0, K, BLOCK_K):
            k_mask = (offs_k + k0) < K
            a = tl.load(x_ptr + k0 + offs_k, mask=k_mask, other=0.0).to(
                tl.float32
            )
            b = tl.load(w_ptrs, mask=n_mask & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc += b * a[None, :]
            w_ptrs += BLOCK_K

    out = tl.sum(acc, axis=1).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_n, out, mask=offs_n < N)


_BLOCK_M = 16
_DOT_BLOCK_N = 16
_DOT_BLOCK_K = 512
_DOT_WARPS = 2
_DOT_STAGES = 4

_SK_WAVES = 2.5

# GEMV path (m == 1): N threshold between the "thin n" and "wide n" configs.
_GEMV_N_WIDE = 1024


def _num_sms():
    """SM count through the framework's portable torch device object."""
    fn = _runtime.torch_device_fn
    return fn.get_device_properties(fn.current_device()).multi_processor_count


def _tiny_n_gemm_generic(x, w, out_dtype):
    """Original generic implementation (kept as the K > 8192 fallback)."""
    m, k = x.shape
    n, _ = w.shape

    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)

    if m == 1:
        if n < _GEMV_N_WIDE:
            block_n, block_k, warps, stages = 4, 2048, 4, 2
        else:
            block_n, block_k, warps, stages = 4, 1024, 2, 2
        block_k = min(block_k, triton.next_power_of_2(max(k, 16)))
        grid = (triton.cdiv(n, block_n),)
        _tiny_n_gemv_kernel[grid](
            x,
            w,
            out,
            n,
            k,
            w.stride(0),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_K=(k % block_k == 0),
            num_warps=warps,
            num_stages=stages,
        )
        return out

    block_k = min(_DOT_BLOCK_K, triton.next_power_of_2(max(k, 16)))
    num_pid_n = triton.cdiv(n, _DOT_BLOCK_N)
    tiles = triton.cdiv(m, _BLOCK_M) * num_pid_n

    split_k = 1
    if tiles < _num_sms():
        block_k = min(256, block_k)
        max_steps = max(triton.cdiv(k, block_k), 1)
        split_k = max(
            1, min(triton.cdiv(int(_num_sms() * _SK_WAVES), tiles), max_steps)
        )
        if tiles * split_k < tiles + 4:
            split_k = (
                1  # a couple of extra programs cannot pay for the workspace
            )

    if split_k > 1:
        k_steps = triton.cdiv(triton.cdiv(k, split_k), block_k)
        part = torch.empty(
            (split_k, m, n), device=x.device, dtype=torch.float32
        )
        grid = (tiles * split_k,)
        _tiny_n_dot_splitk_kernel[grid](
            x,
            w,
            part,
            m,
            n,
            k,
            split_k,
            k_steps,
            x.stride(0),
            w.stride(0),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_DOT_BLOCK_N,
            BLOCK_K=block_k,
            IEEE=(out_dtype == torch.float32),
            num_warps=_DOT_WARPS,
            num_stages=_DOT_STAGES,
        )
        total = m * n
        rgrid = (triton.cdiv(total, 256),)
        _tiny_n_reduce_kernel[rgrid](
            part,
            out,
            total,
            split_k,
            BLOCK=256,
            num_warps=1,
            num_stages=1,
        )
        return out

    grid = (tiles,)
    _tiny_n_dot_kernel[grid](
        x,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        w.stride(0),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_DOT_BLOCK_N,
        BLOCK_K=block_k,
        EVEN_K=(k % block_k == 0),
        EVEN_N=(n % _DOT_BLOCK_N == 0),
        IEEE=(out_dtype == torch.float32),
        num_warps=_DOT_WARPS,
        num_stages=_DOT_STAGES,
    )
    return out


def tiny_n_gemm(x, w, out_dtype):
    m, k = x.shape
    n, k2 = w.shape
    assert k == k2, "tiny_n_gemm: x and w must share the K dimension"

    # Fast path 1: ATen native GEMM (single kernel, transpose in-kernel).
    # bf16 with matching out dtype and N inside the measured-fast native
    # band. fp32-accumulating; the bf16 output rounding matches the
    # reference within the bf16 tolerances (probed worst_ratio <= 0.26).
    if (
        x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and out_dtype == torch.bfloat16
        and n <= _MM_MAX_N
    ):
        return torch.matmul(x, w.t())

    # Fast path 2: full-K single-dot Triton MMA (fp32 accumulator, any
    # float in/out dtype; serves bf16 with N > 512 and every fp32-out /
    # mixed-dtype call). BLOCK_K = next_pow2(K) must fit the DSM budget.
    if (
        k <= _MMA_MAX_BK
        and m >= 1
        and n >= 1
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and w.dtype == x.dtype
    ):
        x = x.contiguous()
        w = w.contiguous()
        out = torch.empty((m, n), device=x.device, dtype=out_dtype)
        block_k = triton.next_power_of_2(max(k, 16))
        grid = (
            triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, _MMA_BLOCK_N),
        )
        _tiny_n_mma_kernel[grid](
            x,
            w,
            out,
            m,
            n,
            k,
            x.stride(0),
            w.stride(0),
            out.stride(0),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_MMA_BLOCK_N,
            BLOCK_K=block_k,
            num_warps=4,
        )
        return out

    # Fallback: original generic kernels (K > 8192 and exotic dtypes).
    return _tiny_n_gemm_generic(x, w, out_dtype)


__all__ = ["tiny_n_gemm"]
