import logging
from typing import Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_CONV1D_GEMM_CONFIGS = runtime.get_tuned_config("conv1d_gemm_v3")
_CONV1D_GENERAL_CONFIGS = runtime.get_tuned_config("conv1d_general")
_CONV1D_DEPTHWISE_CONFIGS = runtime.get_tuned_config("conv1d_depthwise_v5")

_GROUP_SIZE_M = 8


def _single(v: Union[int, Sequence[int]], name: str) -> int:
    if isinstance(v, int):
        return int(v)
    if len(v) != 1:
        raise RuntimeError(f"{name} must be an int or a length-1 tuple")
    return int(v[0])


def _conv_out_dim(
    input_size: int,
    pad_left: int,
    pad_right: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return (
        input_size + pad_left + pad_right - dilation * (kernel - 1) - 1
    ) // stride + 1


def _normalize_padding(
    weight: torch.Tensor,
    stride: int,
    padding: Union[str, int, Tuple[int]],
    dilation: int,
) -> Tuple[int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0
        if padding == "same":
            if stride != 1:
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            kw = weight.shape[2]
            effective_kernel = dilation * (kw - 1) + 1
            total_pad = max(effective_kernel - 1, 0)
            pad_left = total_pad // 2
            return pad_left, total_pad - pad_left
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")

    pad = _single(padding, "padding")
    if pad < 0:
        raise RuntimeError("negative padding is not supported")
    return pad, pad


def _any_requires_grad(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )


def _dtype_id(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    return 3


def _check_conv1d_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: Tuple[int, int],
    dilation: int,
    groups: int,
) -> None:
    if input.dim() not in (2, 3) or weight.dim() != 3:
        raise RuntimeError("flag_dnn conv1d expects 2D/3D input and 3D weight")
    if not input.is_cuda:
        raise NotImplementedError("flag_dnn conv1d only supports CUDA input")
    if weight.device != input.device:
        raise RuntimeError("input and weight must be on the same device")
    if bias is not None and bias.device != input.device:
        raise RuntimeError("input and bias must be on the same device")

    supported_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    if input.dtype not in supported_dtypes:
        raise NotImplementedError(f"Unsupported dtype: {input.dtype}")
    if weight.dtype != input.dtype:
        raise RuntimeError("input and weight must have the same dtype")
    if bias is not None and bias.dtype != input.dtype:
        raise RuntimeError("input and bias must have the same dtype")
    if _any_requires_grad(input, weight, bias):
        raise NotImplementedError("flag_dnn conv1d is forward-only")

    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride <= 0:
        raise RuntimeError("stride must be positive")
    if dilation <= 0:
        raise RuntimeError("dilation must be positive")

    c_in = input.shape[-2]
    c_out, c_per_group, kw = weight.shape
    if c_in <= 0 or c_out <= 0 or c_per_group <= 0 or kw <= 0:
        raise RuntimeError("input and weight dimensions must be non-empty")
    if c_in % groups != 0 or c_out % groups != 0:
        raise RuntimeError("channels must be divisible by groups")
    if c_per_group != c_in // groups:
        raise RuntimeError(
            "weight.shape[1] must match input_channels // groups"
        )
    if padding[0] < 0 or padding[1] < 0:
        raise RuntimeError("negative padding is not supported")
    if bias is not None and (bias.dim() != 1 or bias.numel() != c_out):
        raise RuntimeError(f"bias shape mismatch, expected ({c_out},)")


@libentry()
@libtuner(
    configs=_CONV1D_DEPTHWISE_CONFIGS,
    key=["OL", "C_IN", "KW", "DTYPE_ID"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv1d_depthwise_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XL,
    OL,
    C_IN,
    DTYPE_ID,
    x_stride_n,
    x_stride_c,
    x_stride_l,
    w_stride_o,
    w_stride_k,
    bias_stride,
    y_stride_n,
    y_stride_c,
    y_stride_l,
    STRIDE_W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_W: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    # tl.assume(x_stride_c > 0)
    # tl.assume(x_stride_l > 0)
    # tl.assume(w_stride_o > 0)
    # tl.assume(w_stride_k > 0)
    # tl.assume(y_stride_c > 0)
    # tl.assume(y_stride_l > 0)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_l = offs_l < OL
    mask_c = offs_c < C_IN

    acc = tl.zeros((BLOCK_C, BLOCK_L), dtype=tl.float32)
    x_base = x_ptr + pid_n * x_stride_n
    y_base = y_ptr + pid_n * y_stride_n

    for kw in tl.static_range(0, KW):
        iw = offs_l * STRIDE_W - PAD_LEFT + kw * DIL_W
        valid_l = mask_l & (iw >= 0) & (iw < XL)
        x = tl.load(
            x_base + offs_c[:, None] * x_stride_c + iw[None, :] * x_stride_l,
            mask=mask_c[:, None] & valid_l[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_c * w_stride_o + kw * w_stride_k,
            mask=mask_c,
            other=0.0,
        )
        acc += x * w[:, None]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c * bias_stride, mask=mask_c, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_base + offs_c[:, None] * y_stride_c + offs_l[None, :] * y_stride_l,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_l[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV1D_GEMM_CONFIGS,
    key=["M", "CIN_PER_GROUP", "COUT_PER_GROUP", "KW", "DTYPE_ID"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv1d_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XL,
    OL,
    DTYPE_ID,
    x_stride_n,
    x_stride_c,
    x_stride_l,
    w_stride_o,
    w_stride_i,
    w_stride_k,
    bias_stride,
    y_stride_n,
    y_stride_c,
    y_stride_l,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    # tl.assume(x_stride_c > 0)
    # tl.assume(x_stride_l > 0)
    # tl.assume(w_stride_o > 0)
    # tl.assume(w_stride_i > 0)
    # tl.assume(w_stride_k > 0)
    # tl.assume(y_stride_c > 0)
    # tl.assume(y_stride_l > 0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch = offs_m // OL
    ow = offs_m % OL
    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    k_total: tl.constexpr = CIN_PER_GROUP * KW

    for k0 in range(0, k_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        ci = offs_k // KW
        kw = offs_k % KW
        mask_k = offs_k < k_total
        ic_global = pid_g * CIN_PER_GROUP + ci
        iw = ow[:, None] * STRIDE_W - PAD_LEFT + kw[None, :] * DIL_W
        valid_x = mask_m[:, None] & mask_k[None, :] & (iw >= 0) & (iw < XL)

        x = tl.load(
            x_ptr
            + batch[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + iw * x_stride_l,
            mask=valid_x,
            other=0.0,
        )
        w = tl.load(
            w_ptr
            + oc_global[None, :] * w_stride_o
            + ci[:, None] * w_stride_i
            + kw[:, None] * w_stride_k,
            mask=mask_k[:, None] & mask_oc[None, :],
            other=0.0,
        )
        acc = tl.dot(x, w, acc)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + oc_global * bias_stride, mask=mask_oc, other=0.0
        )
        acc += bias[None, :]

    tl.store(
        y_ptr
        + batch[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + ow[:, None] * y_stride_l,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_oc[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV1D_GENERAL_CONFIGS,
    key=["total_elements", "CIN_PER_GROUP", "KW"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv1d_general_fp64_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    XL,
    OL,
    C_OUT,
    x_stride_n,
    x_stride_c,
    x_stride_l,
    w_stride_o,
    w_stride_i,
    w_stride_k,
    bias_stride,
    y_stride_n,
    y_stride_c,
    y_stride_l,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    ow = offsets % OL
    oc = (offsets // OL) % C_OUT
    batch = offsets // (C_OUT * OL)
    group = oc // COUT_PER_GROUP

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc * bias_stride, mask=mask, other=0.0).to(
            tl.float64
        )

    for kw in tl.static_range(0, KW):
        iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
        valid = mask & (iw >= 0) & (iw < XL)
        for ci in tl.static_range(0, CIN_PER_GROUP):
            ic = group * CIN_PER_GROUP + ci
            x = tl.load(
                x_ptr + batch * x_stride_n + ic * x_stride_c + iw * x_stride_l,
                mask=valid,
                other=0.0,
            ).to(tl.float64)
            weight = tl.load(
                w_ptr + oc * w_stride_o + ci * w_stride_i + kw * w_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float64)
            acc += x * weight

    tl.store(
        y_ptr + batch * y_stride_n + oc * y_stride_c + ow * y_stride_l,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[str, int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    stride_w = _single(stride, "stride")
    dilation_w = _single(dilation, "dilation")
    padding_1d = _normalize_padding(weight, stride_w, padding, dilation_w)
    _check_conv1d_inputs(
        input, weight, bias, stride_w, padding_1d, dilation_w, groups
    )

    is_batched = input.dim() == 3
    n = input.shape[0] if is_batched else 1
    c_in = input.shape[-2]
    x_l = input.shape[-1]
    c_out, _, kw = weight.shape
    pad_left, pad_right = padding_1d
    effective_kernel = dilation_w * (kw - 1) + 1

    if x_l + pad_left + pad_right < effective_kernel:
        raise RuntimeError("kernel size can't be greater than input size")

    out_l = _conv_out_dim(x_l, pad_left, pad_right, dilation_w, kw, stride_w)
    if out_l <= 0:
        raise RuntimeError("computed output size is not positive")

    output_shape = (n, c_out, out_l) if is_batched else (c_out, out_l)
    output = torch.empty(
        output_shape,
        device=input.device,
        dtype=input.dtype,
    )

    x_stride_n = input.stride(0) if is_batched else 0
    x_stride_c = input.stride(1) if is_batched else input.stride(0)
    x_stride_l = input.stride(2) if is_batched else input.stride(1)
    y_stride_n = output.stride(0) if is_batched else 0
    y_stride_c = output.stride(1) if is_batched else output.stride(0)
    y_stride_l = output.stride(2) if is_batched else output.stride(1)
    bias_stride = bias.stride(0) if bias is not None else 0

    cout_per_group = c_out // groups
    cin_per_group = c_in // groups
    m = n * out_l
    is_depthwise = groups == c_in and weight.shape[1] == 1 and c_out == c_in
    dtype_id = _dtype_id(input.dtype)

    with torch_device_fn.device(input.device):
        if is_depthwise and input.dtype != torch.float64:

            def grid_depthwise(meta):
                return (
                    triton.cdiv(out_l, meta["BLOCK_L"]),
                    triton.cdiv(c_in, meta["BLOCK_C"]),
                    n,
                )

            conv1d_depthwise_kernel[grid_depthwise](
                input,
                weight,
                bias if bias is not None else output,
                output,
                x_l,
                out_l,
                c_in,
                dtype_id,
                x_stride_n,
                x_stride_c,
                x_stride_l,
                weight.stride(0),
                weight.stride(2),
                bias_stride,
                y_stride_n,
                y_stride_c,
                y_stride_l,
                stride_w,
                pad_left,
                dilation_w,
                kw,
                HAS_BIAS=bias is not None,
            )
            return output

        if input.dtype == torch.float64:
            total = n * c_out * out_l

            def grid_fp64(meta):
                return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

            conv1d_general_fp64_kernel[grid_fp64](
                input,
                weight,
                bias if bias is not None else output,
                output,
                total,
                x_l,
                out_l,
                c_out,
                x_stride_n,
                x_stride_c,
                x_stride_l,
                weight.stride(0),
                weight.stride(1),
                weight.stride(2),
                bias_stride,
                y_stride_n,
                y_stride_c,
                y_stride_l,
                cout_per_group,
                cin_per_group,
                kw,
                stride_w,
                pad_left,
                dilation_w,
                HAS_BIAS=bias is not None,
            )
            return output

        def grid_gemm(meta):
            return (
                triton.cdiv(m, meta["BLOCK_M"])
                * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                groups,
            )

        conv1d_gemm_kernel[grid_gemm](
            input,
            weight,
            bias if bias is not None else output,
            output,
            m,
            x_l,
            out_l,
            dtype_id,
            x_stride_n,
            x_stride_c,
            x_stride_l,
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            bias_stride,
            y_stride_n,
            y_stride_c,
            y_stride_l,
            cin_per_group,
            cout_per_group,
            kw,
            stride_w,
            pad_left,
            dilation_w,
            HAS_BIAS=bias is not None,
            GROUP_M=_GROUP_SIZE_M,
        )

    return output
