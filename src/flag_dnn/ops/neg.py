import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.type_utils import is_bool_dtype


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("neg"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def neg_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 直接取负
    res = -x

    # 存回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def neg(
    input: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN NEG")

    if is_bool_dtype(input.dtype):
        raise RuntimeError(
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or "
            "`logical_not()` operator instead."
        )

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    out_dtype = input.dtype
    out_shape = input.shape

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert (
            out.shape == out_shape
        ), f"out shape {out.shape} mismatch with input shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        neg_kernel[grid](input, out, n_elements)

    return out
