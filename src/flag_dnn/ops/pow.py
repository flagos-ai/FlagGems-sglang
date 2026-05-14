import logging
from typing import Union, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# if error try :
# res = tl.math.exp(y_f32 * tl.math.log(x_f32))

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.ops.binary import collapse_dims, pad_to_max_dims
from flag_dnn.utils.type_utils import (
    is_bool_dtype,
    is_integral_dtype,
    is_python_bool,
    is_python_int,
)


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def pow_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 向上转型到 float32，防止底层 libdevice 找不到 fp16/bf16 的 pow 签名
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    # res = libdevice.pow(x_f32, y_f32)
    log2_x = tl.math.log2(x_f32)
    res = tl.math.exp2(y_f32 * log2_x)

    # 写回时向下转型回目标数据类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def pow_scalar_exponent_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    exponent_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)

    exp_f32 = tl.cast(exponent_val, tl.float32)
    res = libdevice.pow(x_f32, exp_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def pow_scalar_base_kernel(
    y_ptr,
    out_ptr,
    n_elements,
    base_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask)

    y_f32 = y.to(tl.float32)

    base_f32 = tl.cast(base_val, tl.float32)
    res = libdevice.pow(base_f32, y_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def pow_broadcast_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    # 填充后的 6D 形状
    s1,
    s2,
    s3,
    s4,
    s5,
    # X 的 6D Strides
    sx0,
    sx1,
    sx2,
    sx3,
    sx4,
    sx5,
    # Y 的 6D Strides
    sy0,
    sy1,
    sy2,
    sy3,
    sy4,
    sy5,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 坐标还原（从内向外剥洋葱）
    # 由于做了坍缩，许多 sX 实际上是 1，Triton 编译器遇到 x % 1 或 x // 1 会直接优化掉?
    idx5 = offsets % s5
    rem4 = offsets // s5

    idx4 = rem4 % s4
    rem3 = rem4 // s4

    idx3 = rem3 % s3
    rem2 = rem3 // s3

    idx2 = rem2 % s2
    rem1 = rem2 // s2

    idx1 = rem1 % s1
    idx0 = rem1 // s1

    # 计算物理偏移并加载数据
    x_off = (
        idx0 * sx0
        + idx1 * sx1
        + idx2 * sx2
        + idx3 * sx3
        + idx4 * sx4
        + idx5 * sx5
    )
    y_off = (
        idx0 * sy0
        + idx1 * sy1
        + idx2 * sy2
        + idx3 * sy3
        + idx4 * sy4
        + idx5 * sy5
    )

    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    res = libdevice.pow(x_f32, y_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def pow(
    input: Union[torch.Tensor, int, float],
    exponent: Union[torch.Tensor, int, float],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN POW")

    input_is_tensor = isinstance(input, torch.Tensor)
    exp_is_tensor = isinstance(exponent, torch.Tensor)

    if input_is_tensor and (
        not input.is_contiguous()  # type: ignore[union-attr]
    ):
        assert False, "input must be contiguous."
        input = input.contiguous()

    if exp_is_tensor and (
        not exponent.is_contiguous()  # type: ignore[union-attr]
    ):
        assert False, "exponent must be contiguous."
        exponent = exponent.contiguous()

    if not (input_is_tensor or exp_is_tensor):
        raise TypeError("At least one of input or exponent must be a Tensor")

    if isinstance(exponent, torch.Tensor) and is_bool_dtype(exponent.dtype):
        raise NotImplementedError(
            "flag_dnn pow does not support bool tensor exponent"
        )

    if input_is_tensor and is_python_int(exponent) and exponent < 0:
        if is_integral_dtype(input.dtype):  # type: ignore[union-attr]
            raise RuntimeError(
                "Integers to negative integer powers are not allowed."
            )
    # 确定输出形状与广播
    if input_is_tensor and exp_is_tensor:
        out_shape = torch.broadcast_shapes(
            input.shape,  # type: ignore[union-attr]
            exponent.shape,  # type: ignore[union-attr]
        )
        device = input.device  # type: ignore[union-attr]
    elif input_is_tensor:
        out_shape = input.shape  # type: ignore[union-attr]
        device = input.device  # type: ignore[union-attr]
    else:
        out_shape = exponent.shape  # type: ignore[union-attr]
        device = exponent.device  # type: ignore[union-attr]

    out_dtype = (
        out.dtype if out is not None else torch.result_type(input, exponent)
    )
    if (
        is_python_bool(exponent)
        and input_is_tensor
        and is_bool_dtype(input.dtype)  # type: ignore[union-attr]
    ):
        out_dtype = torch.bool

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=device)
    else:
        assert (
            out.shape == out_shape
        ), f"out shape {out.shape} mismatch with broadcast shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(device):
        if (
            input_is_tensor
            and exp_is_tensor
            and (
                input.shape  # type: ignore[union-attr]
                == exponent.shape  # type: ignore[union-attr]
            )
        ):
            pow_tensor_kernel[grid](input, exponent, out, n_elements)
        elif input_is_tensor and exp_is_tensor:
            input_exp = input.expand(out_shape)  # type: ignore[union-attr]
            exponent_exp = exponent.expand(  # type: ignore[union-attr]
                out_shape
            )

            # 维度坍缩
            c_shape, c_sx, c_sy = collapse_dims(
                out_shape, input_exp.stride(), exponent_exp.stride()
            )

            # 填充到 6 维
            f_shape, f_sx, f_sy = pad_to_max_dims(
                c_shape, c_sx, c_sy, max_dims=6
            )

            pow_broadcast_tensor_kernel[grid](
                input,
                exponent,
                out,
                n_elements,
                *f_shape[1:],  # 传入 s1 到 s5
                *f_sx,  # 传入 sx0 到 sx5
                *f_sy,  # 传入 sy0 到 sy5
            )
        elif input_is_tensor:
            pow_scalar_exponent_kernel[grid](
                input, out, n_elements, float(exponent)
            )
        else:
            pow_scalar_base_kernel[grid](
                exponent, out, n_elements, float(input)
            )

    return out
