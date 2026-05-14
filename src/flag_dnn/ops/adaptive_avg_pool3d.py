import logging
import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
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


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _small_od_oh_ow_meta(od: int, oh: int, ow: int):
    def _pick_block(x: int) -> int:
        if x <= 1:
            return 1
        if x <= 2:
            return 2
        if x <= 4:
            return 4
        return 8

    block_d = _pick_block(od)
    block_h = _pick_block(oh)
    block_w = _pick_block(ow)

    num_warps = 1
    if block_d >= 4 or block_h >= 4 or block_w >= 8:
        num_warps = 2
    if block_d >= 8 or block_h >= 8 or block_w >= 16:
        num_warps = 4

    return block_d, block_h, block_w, num_warps


# =============================================================================
# Global Pooling Kernels (OD == OH == OW == 1)
# =============================================================================


# Small DHW, large NC: batch multiple channels in one block
@libentry()
@triton.jit
def global_avg_pool3d_tiled_nc_kernel(
    x_ptr,
    y_ptr,
    NC,
    DHW,
    ACC_DTYPE: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid = tle.program_id(0)

    offs_nc = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    offs_dhw = tl.arange(0, BLOCK_DHW)

    mask_nc = offs_nc < NC
    mask_dhw = offs_dhw < DHW

    ptrs = x_ptr + offs_nc[:, None] * DHW + offs_dhw[None, :]
    mask = mask_nc[:, None] & mask_dhw[None, :]

    vals = tl.load(ptrs, mask=mask, other=0).to(ACC_DTYPE)
    total = tl.sum(vals, axis=1)
    avg = total / DHW

    tl.store(y_ptr + offs_nc, avg.to(y_ptr.dtype.element_ty), mask=mask_nc)


# Small NC, large DHW: fused single-kernel with last-block finalization
@libentry()
@triton.jit
def global_avg_pool3d_fused_kernel(
    x_ptr,
    y_ptr,
    partial_ptr,
    counter_ptr,
    NC,
    DHW,
    COUNTER_BASE,
    SPLIT_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FINALIZE_BLOCK: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_k = tle.program_id(1)

    if pid_nc >= NC:
        return

    chunk = tl.cdiv(DHW, SPLIT_K)
    start = pid_k * chunk
    end = tl.minimum(start + chunk, DHW)

    x_base = x_ptr + pid_nc * DHW
    offs = tl.arange(0, BLOCK_SIZE)

    total = tl.sum(tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE), axis=0)

    for dhw0 in range(start, end, BLOCK_SIZE):
        idx = dhw0 + offs
        mask = idx < end
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        total += tl.sum(vals, axis=0)

    tl.store(
        partial_ptr + pid_nc * SPLIT_K + pid_k,
        total.to(partial_ptr.dtype.element_ty),
    )

    count = tl.atomic_add(counter_ptr + pid_nc, 1)

    if count == COUNTER_BASE + SPLIT_K - 1:
        p_offs = tl.arange(0, FINALIZE_BLOCK)
        p_mask = p_offs < SPLIT_K
        p_vals = tl.load(
            partial_ptr + pid_nc * SPLIT_K + p_offs,
            mask=p_mask,
            other=0,
        ).to(ACC_DTYPE)
        total_sum = tl.sum(p_vals, axis=0)
        avg_val = total_sum / DHW
        tl.store(y_ptr + pid_nc, avg_val.to(y_ptr.dtype.element_ty))


_GLOBAL_POOL3D_FUSED_CACHE: dict = {}


def _get_global_pool3d_fused_bufs(device, acc_dtype, nc, split_k):
    key = (device.index, str(acc_dtype), nc, split_k)
    state = _GLOBAL_POOL3D_FUSED_CACHE.get(key)
    if state is None:
        partial_buf = torch.empty(nc * split_k, device=device, dtype=acc_dtype)
        counter_buf = torch.zeros(nc, device=device, dtype=torch.int32)
        state = {
            "partial": partial_buf,
            "counter": counter_buf,
            "counter_base": 0,
        }
        _GLOBAL_POOL3D_FUSED_CACHE[key] = state

    partial_buf = state["partial"]
    counter_buf = state["counter"]
    counter_base = state["counter_base"]

    # Avoid a per-call device memset in the hot path.
    # Reset only on rare wraparound.
    max_base = torch.iinfo(torch.int32).max - split_k
    if counter_base > max_base:
        counter_buf.zero_()
        counter_base = 0

    state["counter_base"] = counter_base + split_k
    return partial_buf, counter_buf, counter_base


def _global_large_dhw_small_nc_meta(nc: int, dhw: int):
    target_blocks = 512 if nc <= 8 else 256
    split_k = max(4, _next_power_of_2(max(1, target_blocks // nc)))

    max_split_from_dhw = max(4, dhw // 256)
    split_k = min(split_k, _next_power_of_2(max_split_from_dhw))

    chunk = (dhw + split_k - 1) // split_k
    if chunk >= 4096:
        block_size, num_warps = 512, 8
    elif chunk >= 1024:
        block_size, num_warps = 256, 4
    else:
        block_size, num_warps = 128, 2

    return split_k, block_size, num_warps, 2


def _should_use_global_pool3d_fused(nc: int, dhw: int) -> bool:
    # The plain global kernel launches one program per NC lane. When NC is
    # small and DHW is large, that leaves the GPU severely under-occupied.
    if dhw < 8192:
        return False
    if nc <= 8:
        return True
    return nc <= 16 and dhw >= 131072


def _should_use_native_global_pool3d(
    input: torch.Tensor,
    nc: int,
    dhw: int,
    od: int,
    oh: int,
    ow: int,
) -> bool:
    # Native global pooling is stronger for narrow-channel half-precision cases
    # where Triton's launch/coordination overhead dominates.
    return (
        od == 1
        and oh == 1
        and ow == 1
        and input.dtype in (torch.float16, torch.bfloat16)
        and nc <= 16
        and dhw >= 131072
    )


# Default global pooling kernel with autotuning
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool3d_global"),
    key=["N", "C", "D", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def global_avg_pool3d_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    DHW = D * H * W
    x_base = x_ptr + pid_nc * DHW
    offs = tl.arange(0, BLOCK_SIZE)

    total = tl.sum(tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE), axis=0)

    for dhw0 in range(0, DHW, BLOCK_SIZE):
        idx = dhw0 + offs
        mask = idx < DHW
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        total += tl.sum(vals, axis=0)

    avg_val = total / DHW
    tl.store(y_ptr + pid_nc, avg_val.to(y_ptr.dtype.element_ty))


# =============================================================================
# Small Output Size Kernels (OD * OH * OW <= 8)
# =============================================================================


@libentry()
@triton.jit
def adaptive_avg_pool3d_divisible_small_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    od = tl.arange(0, BLOCK_D)[:, None, None]
    oh = tl.arange(0, BLOCK_H)[None, :, None]
    ow = tl.arange(0, BLOCK_W)[None, None, :]

    out_mask = (od < OD) & (oh < OH) & (ow < OW)

    x_base = x_ptr + pid_nc * D * H * W
    y_base = y_ptr + pid_nc * OD * OH * OW

    start_d = od * K_D
    start_h = oh * K_H
    start_w = ow * K_W

    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kd in range(K_D):
        idx_d = start_d + kd
        for kh in range(K_H):
            idx_h = start_h + kh
            for kw in range(K_W):
                idx_w = start_w + kw
                ptrs = x_base + idx_d * (H * W) + idx_h * W + idx_w
                vals = tl.load(ptrs, mask=out_mask, other=0).to(ACC_DTYPE)
                acc += vals

    out = acc / (K_D * K_H * K_W)
    out_ptrs = y_base + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


@libentry()
@triton.jit
def adaptive_avg_pool3d_general_small_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    MAX_K_D: tl.constexpr,
    MAX_K_H: tl.constexpr,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    od = tl.arange(0, BLOCK_D)[:, None, None]
    oh = tl.arange(0, BLOCK_H)[None, :, None]
    ow = tl.arange(0, BLOCK_W)[None, None, :]

    out_mask = (od < OD) & (oh < OH) & (ow < OW)

    x_base = x_ptr + pid_nc * D * H * W
    y_base = y_ptr + pid_nc * OD * OH * OW

    start_d = (od * D) // OD
    end_d = ((od + 1) * D + OD - 1) // OD
    start_h = (oh * H) // OH
    end_h = ((oh + 1) * H + OH - 1) // OH
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_d = tl.minimum(start_d, D)
    end_d = tl.minimum(end_d, D)
    start_h = tl.minimum(start_h, H)
    end_h = tl.minimum(end_h, H)
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    pool_d = tl.maximum(end_d - start_d, 1)
    pool_h = tl.maximum(end_h - start_h, 1)
    pool_w = tl.maximum(end_w - start_w, 1)
    pool_size = pool_d * pool_h * pool_w

    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kd in range(MAX_K_D):
        idx_d = start_d + kd
        valid_d = idx_d < end_d
        for kh in range(MAX_K_H):
            idx_h = start_h + kh
            valid_h = valid_d & (idx_h < end_h)
            for kw in range(MAX_K_W):
                idx_w = start_w + kw
                valid = out_mask & valid_h & (idx_w < end_w)
                ptrs = x_base + idx_d * (H * W) + idx_h * W + idx_w
                vals = tl.load(ptrs, mask=valid, other=0).to(ACC_DTYPE)
                acc += vals

    out = acc / pool_size.to(ACC_DTYPE)
    out_ptrs = y_base + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


# =============================================================================
# Large Output Size Kernels (with autotuning, 3D grid)
# =============================================================================


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool3d_divisible_large"),
    key=["N", "C", "D", "H", "W", "OD", "OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_avg_pool3d_divisible_large_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_od = tle.program_id(1)
    pid_ohow = tle.program_id(2)

    nc = N * C
    if pid_nc >= nc:
        return

    num_ow = (OW + BLOCK_W - 1) // BLOCK_W
    pid_oh = pid_ohow // num_ow
    pid_ow = pid_ohow % num_ow

    od = pid_od * BLOCK_D + tl.arange(0, BLOCK_D)[:, None, None]
    oh = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)[None, :, None]
    ow = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)[None, None, :]

    out_mask = (od < OD) & (oh < OH) & (ow < OW)

    x_base = x_ptr + pid_nc * D * H * W
    y_base = y_ptr + pid_nc * OD * OH * OW

    start_d = od * K_D
    start_h = oh * K_H
    start_w = ow * K_W

    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kd in range(K_D):
        idx_d = start_d + kd
        for kh in range(K_H):
            idx_h = start_h + kh
            for kw in range(K_W):
                idx_w = start_w + kw
                ptrs = x_base + idx_d * (H * W) + idx_h * W + idx_w
                vals = tl.load(ptrs, mask=out_mask, other=0).to(ACC_DTYPE)
                acc += vals

    out = acc / (K_D * K_H * K_W)
    out_ptrs = y_base + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool3d_general_large"),
    key=["N", "C", "D", "H", "W", "OD", "OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_avg_pool3d_general_large_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    MAX_K_D: tl.constexpr,
    MAX_K_H: tl.constexpr,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_od = tle.program_id(1)
    pid_ohow = tle.program_id(2)

    nc = N * C
    if pid_nc >= nc:
        return

    num_ow = (OW + BLOCK_W - 1) // BLOCK_W
    pid_oh = pid_ohow // num_ow
    pid_ow = pid_ohow % num_ow

    od = pid_od * BLOCK_D + tl.arange(0, BLOCK_D)[:, None, None]
    oh = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)[None, :, None]
    ow = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)[None, None, :]

    out_mask = (od < OD) & (oh < OH) & (ow < OW)

    x_base = x_ptr + pid_nc * D * H * W
    y_base = y_ptr + pid_nc * OD * OH * OW

    start_d = (od * D) // OD
    end_d = ((od + 1) * D + OD - 1) // OD
    start_h = (oh * H) // OH
    end_h = ((oh + 1) * H + OH - 1) // OH
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_d = tl.minimum(start_d, D)
    end_d = tl.minimum(end_d, D)
    start_h = tl.minimum(start_h, H)
    end_h = tl.minimum(end_h, H)
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    pool_d = tl.maximum(end_d - start_d, 1)
    pool_h = tl.maximum(end_h - start_h, 1)
    pool_w = tl.maximum(end_w - start_w, 1)
    pool_size = pool_d * pool_h * pool_w

    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=ACC_DTYPE)

    for kd in range(MAX_K_D):
        idx_d = start_d + kd
        valid_d = idx_d < end_d
        for kh in range(MAX_K_H):
            idx_h = start_h + kh
            valid_h = valid_d & (idx_h < end_h)
            for kw in range(MAX_K_W):
                idx_w = start_w + kw
                valid = out_mask & valid_h & (idx_w < end_w)
                ptrs = x_base + idx_d * (H * W) + idx_h * W + idx_w
                vals = tl.load(ptrs, mask=valid, other=0).to(ACC_DTYPE)
                acc += vals

    out = acc / pool_size.to(ACC_DTYPE)
    out_ptrs = y_base + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


# =============================================================================
# Main Function
# =============================================================================


def adaptive_avg_pool3d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int, int, int]],
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADAPTIVE_AVG_POOL3D (output_size={output_size})")

    if isinstance(output_size, int):
        OD = OH = OW = output_size
    else:
        assert (
            len(output_size) == 3
        ), "output_size must be an int or a tuple of 3 ints"
        OD, OH, OW = output_size

    assert input.ndim in [
        4,
        5,
    ], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        input = input.contiguous()

    N, C, D, H, W = input.shape
    y = torch.empty((N, C, OD, OH, OW), dtype=input.dtype, device=input.device)

    if N == 0 or C == 0 or OD == 0 or OH == 0 or OW == 0:
        return y.squeeze(0) if is_4d else y

    acc_dtype = tl.float64 if input.dtype == torch.float64 else tl.float32

    if OD == D and OH == H and OW == W:
        out = input.clone()
        return out.squeeze(0) if is_4d else out

    div_d = (D % OD) == 0
    div_h = (H % OH) == 0
    div_w = (W % OW) == 0
    is_divisible = div_d and div_h and div_w

    with torch_device_fn.device(input.device):
        NC = N * C
        DHW = D * H * W
        OD_OH_OW = OD * OH * OW

        if _should_use_native_global_pool3d(input, NC, DHW, OD, OH, OW):
            out = F.adaptive_avg_pool3d(input, (OD, OH, OW))
            return out.squeeze(0) if is_4d else out

        if OD == 1 and OH == 1 and OW == 1:
            # Small DHW, large NC: batch channels in a 2D tile
            if DHW <= 512 and NC >= 256:
                block_dhw = _next_power_of_2(DHW)
                if block_dhw < 32:
                    block_dhw = 32

                block_nc = 8 if DHW <= 64 else 4
                num_warps = 2 if DHW <= 64 else 4

                grid = (triton.cdiv(NC, block_nc),)
                global_avg_pool3d_tiled_nc_kernel[grid](
                    input,
                    y,
                    NC,
                    DHW,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_NC=block_nc,
                    BLOCK_DHW=block_dhw,
                    num_warps=num_warps,
                    num_stages=1,
                )
            # Small NC, large DHW: split-K keeps
            # the GPU busy for global pooling.
            elif _should_use_global_pool3d_fused(NC, DHW):
                split_k, block_size, num_warps, num_stages = (
                    _global_large_dhw_small_nc_meta(NC, DHW)
                )
                acc_torch_dtype = (
                    torch.float64
                    if input.dtype == torch.float64
                    else torch.float32
                )
                partial_buf, counter_buf, counter_base = (
                    _get_global_pool3d_fused_bufs(
                        input.device, acc_torch_dtype, NC, split_k
                    )
                )

                grid = (NC, split_k)  # type: ignore[assignment]
                global_avg_pool3d_fused_kernel[grid](
                    input,
                    y,
                    partial_buf,
                    counter_buf,
                    NC,
                    DHW,
                    counter_base,
                    SPLIT_K=split_k,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_SIZE=block_size,
                    FINALIZE_BLOCK=_next_power_of_2(split_k),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            # General case: autotuned global kernel
            else:

                def grid(meta):  # type: ignore[misc]
                    return (NC,)

                global_avg_pool3d_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    D,
                    H,
                    W,
                    ACC_DTYPE=acc_dtype,
                )

        elif OD_OH_OW <= 8:
            block_d, block_h, block_w, num_warps = _small_od_oh_ow_meta(
                OD, OH, OW
            )

            if is_divisible:
                k_d = D // OD
                k_h = H // OH
                k_w = W // OW
                grid = (NC,)
                adaptive_avg_pool3d_divisible_small_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    D,
                    H,
                    W,
                    OD,
                    OH,
                    OW,
                    K_D=k_d,
                    K_H=k_h,
                    K_W=k_w,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_D=block_d,
                    BLOCK_H=block_h,
                    BLOCK_W=block_w,
                    num_warps=num_warps,
                    num_stages=1,
                )
            else:
                max_k_d = _exact_max_window(D, OD)
                max_k_h = _exact_max_window(H, OH)
                max_k_w = _exact_max_window(W, OW)
                grid = (NC,)
                adaptive_avg_pool3d_general_small_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    D,
                    H,
                    W,
                    OD,
                    OH,
                    OW,
                    MAX_K_D=max_k_d,
                    MAX_K_H=max_k_h,
                    MAX_K_W=max_k_w,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_D=block_d,
                    BLOCK_H=block_h,
                    BLOCK_W=block_w,
                    num_warps=num_warps,
                    num_stages=1,
                )

        # Large output: autotuned divisible kernel
        elif is_divisible:
            k_d = D // OD
            k_h = H // OH
            k_w = W // OW

            def grid(meta):  # type: ignore[misc]
                return (
                    NC,
                    triton.cdiv(OD, meta["BLOCK_D"]),
                    triton.cdiv(OH, meta["BLOCK_H"])
                    * triton.cdiv(OW, meta["BLOCK_W"]),
                )

            adaptive_avg_pool3d_divisible_large_kernel[grid](
                input,
                y,
                N,
                C,
                D,
                H,
                W,
                OD,
                OH,
                OW,
                K_D=k_d,
                K_H=k_h,
                K_W=k_w,
                ACC_DTYPE=acc_dtype,
            )

        # Large output: autotuned general kernel
        else:
            max_k_d = _exact_max_window(D, OD)
            max_k_h = _exact_max_window(H, OH)
            max_k_w = _exact_max_window(W, OW)

            def grid(meta):  # type: ignore[misc]
                return (
                    NC,
                    triton.cdiv(OD, meta["BLOCK_D"]),
                    triton.cdiv(OH, meta["BLOCK_H"])
                    * triton.cdiv(OW, meta["BLOCK_W"]),
                )

            adaptive_avg_pool3d_general_large_kernel[grid](
                input,
                y,
                N,
                C,
                D,
                H,
                W,
                OD,
                OH,
                OW,
                MAX_K_D=max_k_d,
                MAX_K_H=max_k_h,
                MAX_K_W=max_k_w,
                ACC_DTYPE=acc_dtype,
            )

    return y.squeeze(0) if is_4d else y
