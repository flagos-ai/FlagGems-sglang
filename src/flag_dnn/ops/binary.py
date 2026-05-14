import logging
from typing import Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.type_utils import (
    is_bool_dtype,
    is_integral_dtype,
    is_python_bool,
    is_python_float,
    is_python_int,
)


logger = logging.getLogger(__name__)


# 维度坍缩：合并连续维度，丢弃 size=1 的维度，将任意 N 维化简为最小维度
def collapse_dims(shape, strides_a, strides_b):
    if not shape:
        return [1], [0], [0]

    c_shape, c_str_a, c_str_b = [], [], []

    # 从内向外 (从右向左) 遍历维度
    for i in reversed(range(len(shape))):
        s = shape[i]

        # 直接丢弃所有大小为 1 的维度，因为它对内存偏移的贡献是 0
        if s == 1:
            continue

        if not c_shape:
            # 初始化最内层维度
            c_shape.append(s)
            c_str_a.append(strides_a[i])
            c_str_b.append(strides_b[i])
        else:
            prev_shape = c_shape[-1]
            # 判断当前维度与前一个维度在内存上是否连续
            # 连续的条件：当前维度的 stride == 前一个维度的 stride * 前一个维度的 size
            is_contig_a = strides_a[i] == c_str_a[-1] * prev_shape
            is_contig_b = strides_b[i] == c_str_b[-1] * prev_shape

            if is_contig_a and is_contig_b:
                # 坍缩，将当前维度乘入上一个维度，stride 保持为最内层 stride
                c_shape[-1] *= s
            else:
                # 无法连续，作为一个新的独立维度加入
                c_shape.append(s)
                c_str_a.append(strides_a[i])
                c_str_b.append(strides_b[i])

    if not c_shape:  # 如果全都是 1
        return [1], [0], [0]

    # 因为是从右向左遍历，最后需要翻转回来
    return c_shape[::-1], c_str_a[::-1], c_str_b[::-1]


def pad_to_max_dims(shape, strides_a, strides_b, max_dims=6):
    # 将坍缩后的 shape/strides 填充到固定的 max_dims，以便传入 Triton
    shape = list(shape)
    strides_a = list(strides_a)
    strides_b = list(strides_b)

    if len(shape) > max_dims:
        raise RuntimeError(f"坍缩后依然超过 {max_dims} 维，Not Support.")

    # 在最外层(左侧)填充 size=1, stride=0
    while len(shape) < max_dims:
        shape.insert(0, 1)
        strides_a.insert(0, 0)
        strides_b.insert(0, 0)

    return shape, strides_a, strides_b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    alpha_val,
    ROUND_MODE: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    if OP_TYPE == "add":
        res = x + alpha_val * y
    elif OP_TYPE == "sub":
        res = x - alpha_val * y
    elif OP_TYPE == "mul":
        res = x * y
    elif OP_TYPE == "div":
        res = x / y
        if ROUND_MODE == 1:
            res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
        elif ROUND_MODE == 2:
            res = tl.math.floor(res)
    elif OP_TYPE == "eq":
        res = x == y
    elif OP_TYPE == "ne":
        res = x != y
    elif OP_TYPE == "lt":
        res = x < y
    elif OP_TYPE == "le":
        res = x <= y
    elif OP_TYPE == "ge":
        res = x >= y
    elif OP_TYPE == "gt":
        res = x > y

    # 结果强制转换回输出张量的目标类型，防止隐式提升导致的错误
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_scalar_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    other_val,
    alpha_val,
    ROUND_MODE: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if OP_TYPE == "add":
        res = x + alpha_val * other_val
    elif OP_TYPE == "sub":
        res = x - alpha_val * other_val
    elif OP_TYPE == "mul":
        res = x * other_val
    elif OP_TYPE == "div":
        res = x / other_val
        if ROUND_MODE == 1:
            res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
        elif ROUND_MODE == 2:
            res = tl.math.floor(res)
    elif OP_TYPE == "eq":
        res = x == other_val
    elif OP_TYPE == "ne":
        res = x != other_val
    elif OP_TYPE == "lt":
        res = x < other_val
    elif OP_TYPE == "le":
        res = x <= other_val
    elif OP_TYPE == "ge":
        res = x >= other_val
    elif OP_TYPE == "gt":
        res = x > other_val

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_broadcast_tensor_kernel(
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
    alpha_val,
    ROUND_MODE: tl.constexpr,
    OP_TYPE: tl.constexpr,
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

    if OP_TYPE == "add":
        res = x + alpha_val * y
    elif OP_TYPE == "sub":
        res = x - alpha_val * y
    elif OP_TYPE == "mul":
        res = x * y
    elif OP_TYPE == "div":
        res = x / y
        # 处理舍入模式 (0: None, 1: trunc, 2: floor)
        if ROUND_MODE == 1:
            # trunc (向零取整): 正数向下取整，负数向上取整
            res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
        elif ROUND_MODE == 2:
            # floor (向下取整)
            res = tl.math.floor(res)
    elif OP_TYPE == "eq":
        res = x == y
    elif OP_TYPE == "ne":
        res = x != y
    elif OP_TYPE == "lt":
        res = x < y
    elif OP_TYPE == "le":
        res = x <= y
    elif OP_TYPE == "ge":
        res = x >= y
    elif OP_TYPE == "gt":
        res = x > y

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


"""
计算类型采用原类型，因此float16，bfloat16可能存在精度问题，尤其div op，可在triton kernel中提升精度计算
"""


def _other_is_bool(other: Union[torch.Tensor, int, float, bool]) -> bool:
    return (
        other.dtype == torch.bool
        if isinstance(other, torch.Tensor)
        else is_python_bool(other)
    )


def _other_is_integral(other: Union[torch.Tensor, int, float, bool]) -> bool:
    if isinstance(other, torch.Tensor):
        return is_integral_dtype(other.dtype)
    return is_python_bool(other) or is_python_int(other)


def _validate_binary_args(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float, bool],
    alpha: Union[int, float],
    op_type: str,
):
    if op_type == "sub" and (
        is_bool_dtype(input.dtype) or _other_is_bool(other)
    ):
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is not "
            "supported. If you are trying to invert a mask, use the `~` or "
            "`logical_not()` operator instead."
        )

    if op_type in ("add", "sub") and is_integral_dtype(input.dtype):
        if is_python_float(alpha):
            raise RuntimeError(
                "For integral input tensors, argument alpha must not be a "
                "floating point number."
            )
        if (
            is_python_bool(alpha)
            and torch.result_type(input, other) != torch.bool
        ):
            raise RuntimeError(
                "Boolean alpha only supported for Boolean results."
            )


def _infer_binary_out_dtype(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float, bool],
    rounding_mode: Optional[str],
    op_type: str,
) -> torch.dtype:
    comparison_ops = ["eq", "ne", "lt", "le", "ge", "gt"]
    if op_type in comparison_ops:
        return torch.bool

    if op_type == "div":
        if rounding_mode is None:
            if is_integral_dtype(input.dtype) and _other_is_integral(other):
                return torch.float32
            return torch.result_type(input, other)
        return torch.result_type(input, other)

    return torch.result_type(input, other)


def binary(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float, bool],
    *,
    alpha: Union[int, float] = 1,
    rounding_mode: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    op_type: str = "",
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN {op_type.upper()})")

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    is_other_tensor = isinstance(other, torch.Tensor)
    out_shape = (
        torch.broadcast_shapes(
            input.shape,
            other.shape,  # type: ignore[union-attr]
        )
        if is_other_tensor
        else input.shape
    )

    mode_idx = 0
    if op_type == "div":
        mode_map = {None: 0, "trunc": 1, "floor": 2}
        if rounding_mode not in mode_map:
            raise RuntimeError(
                f"div expected rounding_mode to be"
                f" one of None, 'trunc', 'floor'"
                f" but found {rounding_mode}"
            )
        mode_idx = mode_map[rounding_mode]
        if (
            rounding_mode is not None
            and is_bool_dtype(input.dtype)
            and _other_is_bool(other)
        ):
            raise NotImplementedError(
                f"\"div_{rounding_mode}_cuda\" not implemented for 'Bool'"
            )

    
    if op_type not in ["add", "sub", "mul", "div", "eq", "ne", "lt", "le", "ge", "gt"]:
        raise RuntimeError(f"Unsupported OP_TYPE={op_type} in binary")

    _validate_binary_args(input, other, alpha, op_type)
    out_dtype = _infer_binary_out_dtype(input, other, rounding_mode, op_type)

    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert (
            out.shape == out_shape
        ), f"out shape {out.shape} mismatch with broadcasted shape {out_shape}"

    n_elements = out.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        if is_other_tensor:
            # 形状一致且连续，走一维 Kernel
            if (
                input.shape == other.shape  # type: ignore[union-attr]
                and input.is_contiguous()
                and other.is_contiguous()  # type: ignore[union-attr]
            ):
                binary_tensor_kernel[grid](
                    input,
                    other,
                    out,
                    n_elements,
                    float(alpha),
                    ROUND_MODE=mode_idx,
                    OP_TYPE=op_type,
                )
            # broadcast
            else:
                # 仅逻辑扩展，不触发显存复制
                in_exp = input.expand(out_shape)
                oth_exp = other.expand(out_shape)  # type: ignore[union-attr]

                # 维度坍缩
                c_shape, c_sx, c_sy = collapse_dims(
                    out_shape, in_exp.stride(), oth_exp.stride()
                )

                # 填充到 6 维
                f_shape, f_sx, f_sy = pad_to_max_dims(
                    c_shape, c_sx, c_sy, max_dims=6
                )

                binary_broadcast_tensor_kernel[grid](
                    input,
                    other,
                    out,
                    n_elements,
                    *f_shape[1:],  # 传入 s1 到 s5
                    *f_sx,  # 传入 sx0 到 sx5
                    *f_sy,  # 传入 sy0 到 sy5
                    float(alpha),
                    ROUND_MODE=mode_idx,
                    OP_TYPE=op_type,
                )
        else:
            other_val: Union[bool, int, float]
            if is_python_bool(other):
                other_val = bool(other)
            elif is_python_int(other):
                other_val = int(other)
            else:
                other_val = float(other)
            binary_scalar_kernel[grid](
                input,
                out,
                n_elements,
                other_val,
                float(alpha),
                ROUND_MODE=mode_idx,
                OP_TYPE=op_type,
            )

    return out
