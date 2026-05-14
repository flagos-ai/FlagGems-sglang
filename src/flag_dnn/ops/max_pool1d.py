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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("max_pool1d"),
    key=["N", "C", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def max_pool1d_kernel(
    x_ptr,
    y_ptr,
    idx_ptr,
    N,
    C,
    W,
    OW,
    pad_w,
    STRIDE_W: tl.constexpr,
    DIL_W: tl.constexpr,
    KERNEL_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OW
    mask = offsets < num_elements

    ow = offsets % OW
    c = (offsets // OW) % C
    n = offsets // (C * OW)

    x_base_idx = n * (C * W) + c * W

    w_start = ow * STRIDE_W - pad_w

    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 使用 tl.static_range 强制编译器在编译期展开循环
    for kw in tl.static_range(KERNEL_W):
        # 编译器会在编译期直接算好这里的偏移常数，优化 dilation=1 的分支
        iw = w_start + kw * DIL_W

        valid = (iw >= 0) & (iw < W)
        load_idx = x_base_idx + iw

        val = tl.load(
            x_ptr + load_idx, mask=mask & valid, other=-float("inf")
        ).to(tl.float32)

        update_mask = val > max_val
        max_val = tl.where(update_mask, val, max_val)

        if RETURN_INDICES:
            current_idx = iw
            max_idx = tl.where(update_mask, current_idx, max_idx)

    tl.store(y_ptr + offsets, max_val.to(x_ptr.dtype.element_ty), mask=mask)

    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx, mask=mask)


def max_pool1d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Optional[Union[int, Tuple[int]]] = None,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(
        f"FLAG_DNN MAX_POOL1D "
        f"(kernel={kernel_size}, "
        f"return_indices={return_indices})"
    )

    def _single(x):
        return (x,) if isinstance(x, int) else tuple(x)

    kernel_size = _single(kernel_size)
    stride = _single(stride) if stride is not None else kernel_size
    padding = _single(padding)
    dilation = _single(dilation)

    assert input.ndim in [2, 3], "Input must be 2D or 3D"
    is_2d = input.ndim == 2
    if is_2d:
        input = input.unsqueeze(0)

    N, C, W = input.shape

    def _out_size(L, pad, dil, k, s, ceil):
        out = (L + 2 * pad - dil * (k - 1) - 1) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OW = _out_size(
        W,
        padding[0],  # type: ignore[index]
        dilation[0],  # type: ignore[index]
        kernel_size[0],  # type: ignore[index]
        stride[0],  # type: ignore[index]
        ceil_mode,
    )

    # ceil_mode 边缘丢弃
    if ceil_mode:
        if (OW - 1) * stride[0] >= W + padding[0]:  # type: ignore[index]
            OW -= 1

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty((N, C, OW), dtype=input.dtype, device=input.device)

    idx = (
        torch.empty((N, C, OW), dtype=torch.int64, device=input.device)
        if return_indices
        else None
    )

    M = N * C * OW
    if M == 0:
        out_y = y.squeeze(0) if is_2d else y
        if return_indices:
            return out_y, (idx.squeeze(0) if is_2d else idx)
        return out_y

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        max_pool1d_kernel[grid](
            input,
            y,
            idx,
            N,
            C,
            W,
            OW,
            padding[0],  # type: ignore[index]
            STRIDE_W=stride[0],  # type: ignore[index]
            DIL_W=dilation[0],  # type: ignore[index]
            KERNEL_W=kernel_size[0],  # type: ignore[index]
            RETURN_INDICES=return_indices,
        )

    out_y = y.squeeze(0) if is_2d else y
    if return_indices:
        return out_y, (idx.squeeze(0) if is_2d else idx)
    return out_y
