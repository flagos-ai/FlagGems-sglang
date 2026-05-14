import logging
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.ops.binary import collapse_dims, pad_to_max_dims
from flag_dnn.utils.type_utils import is_bool_dtype, is_python_bool


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("clamp"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def clamp_kernel(
    x_ptr,
    out_ptr,
    min_ptr,
    max_ptr,  # Tensor 指针
    min_val,
    max_val,  # Scalar 数值
    n_elements,
    HAS_MIN: tl.constexpr,
    HAS_MAX: tl.constexpr,
    IS_MIN_TENSOR: tl.constexpr,
    IS_MAX_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # 统一转换到 float32 进行数值比较，防止低精度截断
    res = x.to(tl.float32)

    # 处理下界 (min)
    if HAS_MIN:
        if IS_MIN_TENSOR:
            min_t = tl.load(min_ptr + offsets, mask=mask).to(tl.float32)
            res = tl.maximum(res, min_t)
        else:
            res = tl.maximum(res, min_val)

    # 处理上界 (max)
    if HAS_MAX:
        if IS_MAX_TENSOR:
            max_t = tl.load(max_ptr + offsets, mask=mask).to(tl.float32)
            res = tl.minimum(res, max_t)
        else:
            res = tl.minimum(res, max_val)

    # 存回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("clamp"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def clamp_broadcast_tensor_kernel(
    x_ptr,
    min_ptr,
    max_ptr,
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
    HAS_MIN: tl.constexpr,
    HAS_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 坐标还原
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
    # 统一转换到 float32 进行数值比较，防止低精度截断
    res = x.to(tl.float32)

    # 处理下界 (min)
    if HAS_MIN:
        min_t = tl.load(min_ptr + y_off, mask=mask).to(tl.float32)
        res = tl.maximum(res, min_t)

    # 处理上界 (max)
    if HAS_MAX:
        max_t = tl.load(max_ptr + y_off, mask=mask).to(tl.float32)
        res = tl.minimum(res, max_t)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def clamp(
    input: torch.Tensor,
    min: Optional[Union[float, int, torch.Tensor]] = None,
    max: Optional[Union[float, int, torch.Tensor]] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN CLAMP")

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    if min is None and max is None:
        raise RuntimeError("At least one of 'min' or 'max' must not be None")

    has_min = min is not None
    has_max = max is not None
    is_min_tensor = isinstance(min, torch.Tensor)
    is_max_tensor = isinstance(max, torch.Tensor)

    if is_bool_dtype(input.dtype):
        if (has_min and not is_min_tensor and is_python_bool(min)) or (
            has_max and not is_max_tensor and is_python_bool(max)
        ):
            raise NotImplementedError(
                "flag_dnn clamp does not support bool scalar bounds "
                f"for dtype={input.dtype}"
            )

    if has_min and has_max and is_min_tensor != is_max_tensor:
        raise RuntimeError(
            "'min' and 'max' must be equal(Number or Tensor) at the same time"
        )

    if (
        has_min
        and has_max
        and is_min_tensor
        and is_max_tensor
        and min.shape != max.shape  # type: ignore[union-attr]
    ):
        raise RuntimeError(
            "Not supported when 'min' and 'max'"
            " are both Tensors but with"
            " different shapes"
        )

    need_broadcast = False
    if (
        has_min
        and is_min_tensor
        and input.shape != min.shape  # type: ignore[union-attr]
    ):
        need_broadcast = True
    if (
        has_max
        and is_max_tensor
        and input.shape != max.shape  # type: ignore[union-attr]
    ):
        need_broadcast = True

    # 形状推导 (Broadcasting)
    # 动态计算 input, min, max 广播后的最终全局形状
    out_shape = input.shape
    if need_broadcast and has_min:
        out_shape = torch.broadcast_shapes(
            out_shape,
            min.shape,  # type: ignore[union-attr]
        )
    if need_broadcast and has_max:
        out_shape = torch.broadcast_shapes(
            out_shape,
            max.shape,  # type: ignore[union-attr]
        )

    out_dtype = input.dtype
    if has_min:
        out_dtype = torch.result_type(input, min)
    if has_max:
        out_dtype = torch.result_type(
            torch.empty((), dtype=out_dtype, device=input.device),
            max,
        )

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert (
            out.shape == out_shape
        ), f"out shape {out.shape} mismatch with broadcasted shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # 处理 min 指针与常量
    min_val = 0.0
    if has_min:
        if is_min_tensor and (
            input.shape != min.shape  # type: ignore[union-attr]
        ):
            input_exp = input.expand(out_shape)
            min_exp = min.expand(out_shape)  # type: ignore[union-attr]
            min_c_shape, min_c_sx, min_c_sy = collapse_dims(
                out_shape, input_exp.stride(), min_exp.stride()
            )
            f_shape, f_sx, f_sy = pad_to_max_dims(
                min_c_shape, min_c_sx, min_c_sy, max_dims=6
            )
        elif not is_min_tensor:
            min_val = float(min)

    # 处理 max 指针与常量
    max_val = 0.0
    if has_max:
        if is_max_tensor and (
            input.shape != max.shape  # type: ignore[union-attr]
        ):
            input_exp = input.expand(out_shape)
            max_exp = max.expand(out_shape)  # type: ignore[union-attr]
            max_c_shape, max_c_sx, max_c_sy = collapse_dims(
                out_shape, input_exp.stride(), max_exp.stride()
            )
            f_shape, f_sx, f_sy = pad_to_max_dims(
                max_c_shape, max_c_sx, max_c_sy, max_dims=6
            )
        elif not is_max_tensor:
            max_val = float(max)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        if not need_broadcast:
            clamp_kernel[grid](
                input,
                out,
                min,
                max,
                min_val,
                max_val,
                n_elements,
                HAS_MIN=has_min,
                HAS_MAX=has_max,
                IS_MIN_TENSOR=is_min_tensor,
                IS_MAX_TENSOR=is_max_tensor,
            )
        else:
            clamp_broadcast_tensor_kernel[grid](
                input,
                min,
                max,
                out,
                n_elements,
                *f_shape[1:],  # 传入 s1 到 s5
                *f_sx,  # 传入 sx0 到 sx5
                *f_sy,  # 传入 sy0 到 sy5
                HAS_MIN=has_min,
                HAS_MAX=has_max,
            )

    return out
