import logging
import math
from typing import Tuple, Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _exact_max_window(in_size: int, out_size: int) -> int:
    gcd_val = math.gcd(in_size, out_size)
    max_r = out_size - gcd_val
    return (max_r + in_size + out_size - 1) // out_size


def _small_oh_ow_meta(oh: int, ow: int):
    def _pick_block(x: int) -> int:
        if x <= 1:
            return 1
        if x <= 2:
            return 2
        if x <= 4:
            return 4
        return 8

    block_h = _pick_block(oh)
    block_w = _pick_block(ow)
    return block_h, block_w, 1


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


_GLOBAL_POOL2D_PARTIAL_CACHE: dict = {}


def _get_global_pool2d_partial(device, dtype, nc, split_k):
    key = (device.index, str(dtype), nc, split_k)
    buf = _GLOBAL_POOL2D_PARTIAL_CACHE.get(key)
    if buf is None or buf.shape != (nc, split_k):
        buf = torch.empty((nc, split_k), device=device, dtype=dtype)
        _GLOBAL_POOL2D_PARTIAL_CACHE[key] = buf
    return buf


def _global_large_hw_small_nc_meta(nc: int, hw: int):
    if nc <= 64 and hw >= 32768:
        return 8, 512, 8, 2  # split_k, block_size, num_warps, num_stages
    elif nc <= 64 and hw >= 8192:
        return 8, 256, 4, 2
    else:
        return 4, 256, 4, 2


@libentry()
@triton.jit
def global_avg_pool2d_tiled_nc_kernel(
    x_ptr,
    y_ptr,
    NC,
    HW,
    ACC_DTYPE: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tle.program_id(0)

    offs_nc = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    offs_hw = tl.arange(0, BLOCK_HW)

    mask_nc = offs_nc < NC
    mask_hw = offs_hw < HW

    ptrs = x_ptr + offs_nc[:, None] * HW + offs_hw[None, :]
    mask = mask_nc[:, None] & mask_hw[None, :]

    vals = tl.load(ptrs, mask=mask, other=0).to(ACC_DTYPE)
    total = tl.sum(vals, axis=1)
    avg = total / HW

    tl.store(y_ptr + offs_nc, avg.to(y_ptr.dtype.element_ty), mask=mask_nc)


@triton.jit
def global_avg_pool2d_split_partial_kernel(
    x_ptr,
    partial_ptr,
    NC,
    HW,
    SPLIT_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_k = tle.program_id(1)

    if pid_nc >= NC:
        return

    chunk = tl.cdiv(HW, SPLIT_K)
    start = pid_k * chunk
    end = tl.minimum(start + chunk, HW)

    x_base = x_ptr + pid_nc * HW
    offs = tl.arange(0, BLOCK_SIZE)

    total = tl.sum(tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE), axis=0)

    for hw0 in range(start, end, BLOCK_SIZE):
        idx = hw0 + offs
        mask = idx < end
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        total += tl.sum(vals, axis=0)

    tl.store(partial_ptr + pid_nc * SPLIT_K + pid_k, total)


@triton.jit
def global_avg_pool2d_split_finalize_kernel(
    partial_ptr,
    y_ptr,
    NC,
    HW,
    SPLIT_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    if pid_nc >= NC:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < SPLIT_K

    vals = tl.load(
        partial_ptr + pid_nc * SPLIT_K + offs, mask=mask, other=0
    ).to(ACC_DTYPE)

    total = tl.sum(vals, axis=0)
    avg_val = total / HW

    tl.store(y_ptr + pid_nc, avg_val.to(y_ptr.dtype.element_ty))


# OH == 1 and OW == 1
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool2d_global"),
    key=["N", "C", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def global_avg_pool2d_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    HW = H * W
    x_base = x_ptr + pid_nc * HW
    offs = tl.arange(0, BLOCK_SIZE)

    total = tl.sum(tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE), axis=0)

    for hw0 in range(0, HW, BLOCK_SIZE):
        idx = hw0 + offs
        mask = idx < HW
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        total += tl.sum(vals, axis=0)

    avg_val = total / HW
    tl.store(y_ptr + pid_nc, avg_val.to(y_ptr.dtype.element_ty))


# small OH/OW, divisible
@libentry()
@triton.jit
def adaptive_avg_pool2d_divisible_small_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    oh = tl.arange(0, BLOCK_H)[:, None]
    ow = tl.arange(0, BLOCK_W)[None, :]

    oh_mask = oh < OH
    ow_mask = ow < OW
    out_mask = oh_mask & ow_mask

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * H * W
    y_base = y_ptr + (n * C + c) * OH * OW

    start_h = oh * K_H
    start_w = ow * K_W

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kh in range(K_H):
        idx_h = start_h + kh
        for kw in range(K_W):
            idx_w = start_w + kw
            ptrs = x_base + idx_h * W + idx_w
            vals = tl.load(ptrs, mask=out_mask, other=0).to(ACC_DTYPE)
            acc += vals

    out = acc / (K_H * K_W)
    out_ptrs = y_base + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


# small OH/OW, general
@libentry()
@triton.jit
def adaptive_avg_pool2d_general_small_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    MAX_K_H: tl.constexpr,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    oh = tl.arange(0, BLOCK_H)[:, None]
    ow = tl.arange(0, BLOCK_W)[None, :]

    oh_mask = oh < OH
    ow_mask = ow < OW
    out_mask = oh_mask & ow_mask

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * H * W
    y_base = y_ptr + (n * C + c) * OH * OW

    start_h = (oh * H) // OH
    end_h = ((oh + 1) * H + OH - 1) // OH
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_h = tl.minimum(start_h, H)
    end_h = tl.minimum(end_h, H)
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    pool_h = tl.maximum(end_h - start_h, 1)
    pool_w = tl.maximum(end_w - start_w, 1)
    pool_size = pool_h * pool_w

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kh in range(MAX_K_H):
        idx_h = start_h + kh
        valid_h = idx_h < end_h
        for kw in range(MAX_K_W):
            idx_w = start_w + kw
            valid = out_mask & valid_h & (idx_w < end_w)
            ptrs = x_base + idx_h * W + idx_w
            vals = tl.load(ptrs, mask=valid, other=0).to(ACC_DTYPE)
            acc += vals

    out = acc / pool_size.to(ACC_DTYPE)
    out_ptrs = y_base + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


# large OH/OW, divisible
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool2d_divisible_large"),
    key=["N", "C", "H", "W", "OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_avg_pool2d_divisible_large_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_oh = tle.program_id(1)
    pid_ow = tle.program_id(2)

    nc = N * C
    if pid_nc >= nc:
        return

    oh = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    oh_mask = oh < OH
    ow_mask = ow < OW
    out_mask = oh_mask & ow_mask

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * H * W
    y_base = y_ptr + (n * C + c) * OH * OW

    start_h = oh * K_H
    start_w = ow * K_W

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kh in range(K_H):
        idx_h = start_h + kh
        for kw in range(K_W):
            idx_w = start_w + kw
            ptrs = x_base + idx_h * W + idx_w
            vals = tl.load(ptrs, mask=out_mask, other=0).to(ACC_DTYPE)
            acc += vals

    out = acc / (K_H * K_W)
    out_ptrs = y_base + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


# large OH/OW, general
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool2d_general_large"),
    key=["N", "C", "H", "W", "OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_avg_pool2d_general_large_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    MAX_K_H: tl.constexpr,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_oh = tle.program_id(1)
    pid_ow = tle.program_id(2)

    nc = N * C
    if pid_nc >= nc:
        return

    oh = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    oh_mask = oh < OH
    ow_mask = ow < OW
    out_mask = oh_mask & ow_mask

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * H * W
    y_base = y_ptr + (n * C + c) * OH * OW

    start_h = (oh * H) // OH
    end_h = ((oh + 1) * H + OH - 1) // OH
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_h = tl.minimum(start_h, H)
    end_h = tl.minimum(end_h, H)
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    pool_h = tl.maximum(end_h - start_h, 1)
    pool_w = tl.maximum(end_w - start_w, 1)
    pool_size = pool_h * pool_w

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kh in range(MAX_K_H):
        idx_h = start_h + kh
        valid_h = idx_h < end_h
        for kw in range(MAX_K_W):
            idx_w = start_w + kw
            valid = out_mask & valid_h & (idx_w < end_w)
            ptrs = x_base + idx_h * W + idx_w
            vals = tl.load(ptrs, mask=valid, other=0).to(ACC_DTYPE)
            acc += vals

    out = acc / pool_size.to(ACC_DTYPE)
    out_ptrs = y_base + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


def adaptive_avg_pool2d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[Optional[int], Optional[int]]],
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADAPTIVE_AVG_POOL2D (output_size={output_size})")

    if isinstance(output_size, int):
        OH = output_size
        OW = output_size
    else:
        assert (
            len(output_size) == 2
        ), "output_size for adaptive_avg_pool2d must have 2 elements"
        OH = output_size[0] if output_size[0] is not None else input.shape[-2]
        OW = output_size[1] if output_size[1] is not None else input.shape[-1]

    assert input.ndim in [3, 4], "Input must be 3D or 4D"
    is_3d = input.ndim == 3
    if is_3d:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    N, C, H, W = input.shape
    y = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    if N == 0 or C == 0 or OH == 0 or OW == 0:
        return y.squeeze(0) if is_3d else y

    acc_dtype = tl.float64 if input.dtype == torch.float64 else tl.float32

    # identity fast path
    if OH == H and OW == W:
        out = input.clone()
        return out.squeeze(0) if is_3d else out

    div_h = (H % OH) == 0
    div_w = (W % OW) == 0
    is_divisible = div_h and div_w

    with torch_device_fn.device(input.device):
        # global avg
        if OH == 1 and OW == 1:
            NC = N * C
            HW = H * W
            # 1) HW 小、NC 大：一个 program 算多个 channel
            if HW <= 256 and NC >= 256:
                block_hw = _next_power_of_2(HW)
                if block_hw < 32:
                    block_hw = 32

                if HW <= 64:
                    block_nc = 8
                    num_warps = 2
                else:
                    block_nc = 4
                    num_warps = 4

                grid = (triton.cdiv(NC, block_nc),)
                global_avg_pool2d_tiled_nc_kernel[grid](
                    input,
                    y,
                    NC,
                    HW,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_NC=block_nc,
                    BLOCK_HW=block_hw,
                    num_warps=num_warps,
                    num_stages=1,
                )
            # 2) NC 很小但 HW 很大：split reduction
            elif NC <= 64 and HW >= 8192:
                split_k, block_size, num_warps, num_stages = (
                    _global_large_hw_small_nc_meta(NC, HW)
                )
                partial_dtype = (
                    torch.float64
                    if input.dtype == torch.float64
                    else torch.float32
                )
                partial = _get_global_pool2d_partial(
                    input.device, partial_dtype, NC, split_k
                )

                grid = (NC, split_k)  # type: ignore[assignment]
                global_avg_pool2d_split_partial_kernel[grid](
                    input,
                    partial,
                    NC,
                    HW,
                    SPLIT_K=split_k,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

                grid = (NC,)
                global_avg_pool2d_split_finalize_kernel[grid](
                    partial,
                    y,
                    NC,
                    HW,
                    SPLIT_K=split_k,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_SIZE=8 if split_k <= 8 else 16,
                    num_warps=1,
                    num_stages=1,
                )
            # 3) 其余走原来的 global kernel
            else:

                def grid(meta):  # type: ignore[misc]
                    return (NC,)

                global_avg_pool2d_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    H,
                    W,
                    ACC_DTYPE=acc_dtype,
                )
        # small OH/OW: fixed small tile, no autotune
        elif OH * OW <= 16:
            block_h, block_w, num_warps = _small_oh_ow_meta(OH, OW)

            if is_divisible:
                k_h = H // OH
                k_w = W // OW
                grid = (N * C,)
                adaptive_avg_pool2d_divisible_small_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    H,
                    W,
                    OH,
                    OW,
                    K_H=k_h,
                    K_W=k_w,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_H=block_h,
                    BLOCK_W=block_w,
                    num_warps=num_warps,
                    num_stages=1,
                )
            else:
                max_k_h = _exact_max_window(H, OH)
                max_k_w = _exact_max_window(W, OW)
                grid = (N * C,)
                adaptive_avg_pool2d_general_small_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    H,
                    W,
                    OH,
                    OW,
                    MAX_K_H=max_k_h,
                    MAX_K_W=max_k_w,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_H=block_h,
                    BLOCK_W=block_w,
                    num_warps=num_warps,
                    num_stages=1,
                )
        # large OH/OW: divisible
        elif is_divisible:
            k_h = H // OH
            k_w = W // OW

            def grid(meta):  # type: ignore[misc]
                return (
                    N * C,
                    triton.cdiv(OH, meta["BLOCK_H"]),
                    triton.cdiv(OW, meta["BLOCK_W"]),
                )

            adaptive_avg_pool2d_divisible_large_kernel[grid](
                input,
                y,
                N,
                C,
                H,
                W,
                OH,
                OW,
                K_H=k_h,
                K_W=k_w,
                ACC_DTYPE=acc_dtype,
            )
        # large OH/OW: general
        else:
            max_k_h = _exact_max_window(H, OH)
            max_k_w = _exact_max_window(W, OW)

            def grid(meta):  # type: ignore[misc]
                return (
                    N * C,
                    triton.cdiv(OH, meta["BLOCK_H"]),
                    triton.cdiv(OW, meta["BLOCK_W"]),
                )

            adaptive_avg_pool2d_general_large_kernel[grid](
                input,
                y,
                N,
                C,
                H,
                W,
                OH,
                OW,
                MAX_K_H=max_k_h,
                MAX_K_W=max_k_w,
                ACC_DTYPE=acc_dtype,
            )

    return y.squeeze(0) if is_3d else y
