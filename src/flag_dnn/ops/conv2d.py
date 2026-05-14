import ctypes
import ctypes.util
import logging
import threading
import weakref
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Tuning spaces.
_CONV2D_SPATIAL_CONFIGS = runtime.get_tuned_config("conv2d_spatial")
_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS = runtime.get_tuned_config(
    "conv2d_spatial_nchw_packed"
)
_CONV2D_1X1_CONFIGS = runtime.get_tuned_config("conv2d_1x1")
_CONV2D_1X1_NCHW_M_CONFIGS = runtime.get_tuned_config("conv2d_1x1_nchw_m")
_DW_CONV2D_V2_CONFIGS = runtime.get_tuned_config("conv2d_dw_v2")
_DW_CONV2D_C1_CONFIGS = runtime.get_tuned_config("conv2d_dw_c1")

# Small LRU cache for packed weights.  The cache verifies tensor identity with a
# weakref, so a later tensor that happens to reuse the same data_ptr cannot get a
# stale packed weight.
_PACKED_WEIGHT_CACHE: "OrderedDict[tuple, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]]" = OrderedDict()
_PACKED_WEIGHT_CACHE_MAX = 32

# Grouped program ordering generally improves
# L2 reuse for implicit-GEMM kernels.
_GROUP_SIZE_M = 8


# Optional low-level cuBLAS bridge for FP64 GEMM-backed paths.  This avoids
# torch.mm / torch.matmul while still using NVIDIA's optimized DGEMM for cases
# where pure Triton fp64 direct convolution is not competitive.  If cuBLAS is
# unavailable, the code falls back to the pure Triton fp64 kernels below.
_CUBLAS_OP_N = 0
_CUBLAS_STATUS_SUCCESS = 0
_CUBLAS_HANDLE_CACHE = {}
_CUBLAS_LOCK = threading.Lock()
_CUBLAS_LIB = None
_USE_CUBLAS_FP64 = True


def _load_cublas_lib():
    global _CUBLAS_LIB
    if _CUBLAS_LIB is not None:
        return _CUBLAS_LIB

    names = []
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(("libcublas.so.12", "libcublas.so.11", "libcublas.so"))

    last_err = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.cublasCreate_v2.restype = ctypes.c_int
            lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.cublasSetStream_v2.restype = ctypes.c_int
            lib.cublasDgemmStridedBatched.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_int,  # transa
                ctypes.c_int,  # transb
                ctypes.c_int,  # m
                ctypes.c_int,  # n
                ctypes.c_int,  # k
                ctypes.POINTER(ctypes.c_double),  # alpha
                ctypes.c_void_p,  # A
                ctypes.c_int,  # lda
                ctypes.c_longlong,  # strideA
                ctypes.c_void_p,  # B
                ctypes.c_int,  # ldb
                ctypes.c_longlong,  # strideB
                ctypes.POINTER(ctypes.c_double),  # beta
                ctypes.c_void_p,  # C
                ctypes.c_int,  # ldc
                ctypes.c_longlong,  # strideC
                ctypes.c_int,  # batchCount
            ]
            lib.cublasDgemmStridedBatched.restype = ctypes.c_int
            _CUBLAS_LIB = lib
            return lib
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            last_err = exc

    logger.debug("cuBLAS could not be loaded for fp64 conv2d fast path: %s", last_err)
    return None


def _device_index(device: torch.device) -> int:
    return 0 if device.index is None else int(device.index)


def _get_cublas_handle(device: torch.device):
    lib = _load_cublas_lib()
    if lib is None:
        return None, None

    dev_idx = _device_index(device)
    with _CUBLAS_LOCK:
        handle = _CUBLAS_HANDLE_CACHE.get(dev_idx)
        if handle is None:
            handle = ctypes.c_void_p()
            status = lib.cublasCreate_v2(ctypes.byref(handle))
            if status != _CUBLAS_STATUS_SUCCESS:
                logger.debug("cublasCreate_v2 failed with status %s", status)
                return None, None
            _CUBLAS_HANDLE_CACHE[dev_idx] = handle
    return lib, handle


def _cublas_dgemm_strided_batched(
    ref_tensor: torch.Tensor,
    a_ptr: int,
    b_ptr: int,
    c_ptr: int,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    stride_a: int,
    stride_b: int,
    stride_c: int,
    batch_count: int,
    beta_value: float = 0.0,
) -> bool:
    if batch_count <= 0 or m <= 0 or n <= 0 or k <= 0:
        return True

    lib, handle = _get_cublas_handle(ref_tensor.device)
    if lib is None or handle is None:
        return False

    stream = torch.cuda.current_stream(ref_tensor.device).cuda_stream
    status = lib.cublasSetStream_v2(handle, ctypes.c_void_p(int(stream)))
    if status != _CUBLAS_STATUS_SUCCESS:
        logger.debug("cublasSetStream_v2 failed with status %s", status)
        return False

    alpha = ctypes.c_double(1.0)
    beta = ctypes.c_double(beta_value)
    status = lib.cublasDgemmStridedBatched(
        handle,
        _CUBLAS_OP_N,
        _CUBLAS_OP_N,
        int(m),
        int(n),
        int(k),
        ctypes.byref(alpha),
        ctypes.c_void_p(int(a_ptr)),
        int(lda),
        ctypes.c_longlong(int(stride_a)),
        ctypes.c_void_p(int(b_ptr)),
        int(ldb),
        ctypes.c_longlong(int(stride_b)),
        ctypes.byref(beta),
        ctypes.c_void_p(int(c_ptr)),
        int(ldc),
        ctypes.c_longlong(int(stride_c)),
        int(batch_count),
    )
    if status != _CUBLAS_STATUS_SUCCESS:
        logger.debug("cublasDgemmStridedBatched failed with status %s", status)
        return False
    return True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _pair(v: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(v, int):
        return v, v
    if len(v) != 2:
        raise RuntimeError(f"expected length 2, but got {v}")
    return int(v[0]), int(v[1])


def _conv_out_dim(
    input_size: int,
    pad_before: int,
    pad_after: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return (
        input_size + pad_before + pad_after - dilation * (kernel - 1) - 1
    ) // stride + 1


def _normalize_padding(
    weight: torch.Tensor,
    stride: Tuple[int, int],
    padding: Union[str, int, Tuple[int, int]],
    dilation: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0, 0, 0)
        if padding == "same":
            if stride != (1, 1):
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            kh, kw = weight.shape[2], weight.shape[3]
            dil_h, dil_w = dilation
            eff_kh, eff_kw = dil_h * (kh - 1) + 1, dil_w * (kw - 1) + 1
            pad_h, pad_w = max(eff_kh - 1, 0), max(eff_kw - 1, 0)
            pad_top, pad_left = pad_h // 2, pad_w // 2
            return (pad_top, pad_h - pad_top, pad_left, pad_w - pad_left)
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")
    pad_h, pad_w = _pair(padding)
    return (pad_h, pad_h, pad_w, pad_w)


def _check_conv2d_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> None:
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError("flag_dnn conv2d expects 4D input and weight")
    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(f"Unsupported dtype: {input.dtype}")
    if weight.dtype != input.dtype:
        raise RuntimeError("input and weight must have the same dtype")
    if input.device != weight.device:
        raise RuntimeError("input and weight must be on the same device")
    if bias is not None:
        if bias.dtype != input.dtype:
            raise RuntimeError("bias must have the same dtype as input")
        if bias.device != input.device:
            raise RuntimeError("bias must be on the same device as input")
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride[0] <= 0 or stride[1] <= 0:
        raise RuntimeError("stride must be positive")
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError("dilation must be positive")
    if min(padding) < 0:
        raise RuntimeError("negative padding is not supported")

    _, c_in, _, _ = input.shape
    c_out, c_per_group, _, _ = weight.shape
    if c_in % groups != 0 or c_out % groups != 0:
        raise RuntimeError("channels must be divisible by groups")
    if c_per_group != c_in // groups:
        raise RuntimeError("weight.shape[1] must match input_channels // groups")
    if bias is not None and (bias.dim() != 1 or bias.numel() != c_out):
        raise RuntimeError(f"bias shape mismatch, expected ({c_out},)")


def _dtype_id(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float64:
        return 3
    return -1


def _input_has_fast_channel_stride(x: torch.Tensor) -> bool:
    return x.dim() == 4 and x.stride(1) == 1


def _cache_get_or_create(
    key: tuple,
    owner: torch.Tensor,
    fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    entry = _PACKED_WEIGHT_CACHE.get(key)
    if entry is not None:
        owner_ref, value = entry
        if owner_ref() is owner:
            _PACKED_WEIGHT_CACHE.move_to_end(key)
            return value
        # Avoid stale cache hits if Python/Torch reused an old data_ptr/id.
        del _PACKED_WEIGHT_CACHE[key]

    value = fn()
    _PACKED_WEIGHT_CACHE[key] = (weakref.ref(owner), value)
    _PACKED_WEIGHT_CACHE.move_to_end(key)
    while len(_PACKED_WEIGHT_CACHE) > _PACKED_WEIGHT_CACHE_MAX:
        _PACKED_WEIGHT_CACHE.popitem(last=False)
    return value


def _weight_cache_key(tag: str, weight: torch.Tensor, groups: int) -> tuple:
    # `_version` increments on in-place writes and invalidates the packed copy.
    version = int(getattr(weight, "_version", 0))
    return (
        tag,
        id(weight),
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        str(weight.dtype),
        weight.device.type,
        weight.device.index,
        groups,
        version,
    )


def _pack_depthwise_weight_khw_c(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [C, 1, KH, KW] -> [KH, KW, C]
    key = _weight_cache_key("depthwise_khw_c", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c, _, kh, kw = base.shape
        return base.view(c, kh, kw).permute(1, 2, 0).contiguous()

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_1x1_nchw(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, 1, 1] -> [G, CoutG, CinG]
    key = _weight_cache_key("1x1_nchw", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, _, _ = base.shape
        cout_g = c_out // groups
        return base.view(groups, cout_g, cin_g)

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_1x1_cl(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, 1, 1] -> [G, CinG, CoutG]
    key = _weight_cache_key("1x1_cl", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, _, _ = base.shape
        cout_g = c_out // groups
        return base.view(groups, cout_g, cin_g).permute(0, 2, 1).contiguous()

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_spatial_cl(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, KH, KW] -> [G, CinG, KH, KW, CoutG]
    #
    # conv2d_spatial_cl_kernel uses offs_k flattened as:
    #   ic * KH * KW + kh * KW + kw
    # so the packed weight must use [CinG, KH, KW, CoutG] order inside each group.
    key = _weight_cache_key("spatial_cl_cin_khw", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, kh, kw = base.shape
        cout_g = c_out // groups
        return (
            base.view(groups, cout_g, cin_g, kh, kw)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
        )

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_spatial_nchw_khw_oci(
    weight: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    # [Cout, CinG, KH, KW] -> [G, KH, KW, CoutG, CinG]
    #
    # Used by the optimized NCHW packed-KHW spatial kernel.
    # The kernel loops kh/kw statically and performs tl.dot over CinG only,
    # avoiding div/mod by KH*KW inside the hot K loop.
    key = _weight_cache_key("spatial_nchw_khw_oci", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, kh, kw = base.shape
        cout_g = c_out // groups
        return (
            base.view(groups, cout_g, cin_g, kh, kw)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    return _cache_get_or_create(key, weight, _fn)


def _use_packed_spatial_nchw(
    dtype: torch.dtype,
    groups: int,
    kh: int,
    kw: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    cin_per_group: int,
    cout_per_group: int,
    oh: int,
    ow: int,
) -> bool:
    if groups != 1:
        return False
    if kh != 3 or kw != 3:
        return False
    if stride != (1, 1):
        return False

    hw = oh * ow

    # fp32 weak shapes:
    #   [32,64,56,56]  x [64,64,3,3]
    #   [16,64,56,56]  x [64,64,3,3], dilation=2
    #   [16,128,28,28] x [128,128,3,3], dilation=4
    if dtype == torch.float32 and cin_per_group >= 64 and cout_per_group >= 64:
        return True

    # fp16/bf16 weak large-channel 3x3 family.
    return (
        dilation == (1, 1)
        and cin_per_group >= 128
        and cout_per_group >= 128
        and hw <= 28 * 28
    )


def _use_1x1_nchw_m_kernel(
    dtype: torch.dtype,
    groups: int,
    kh: int,
    kw: int,
    dilation: Tuple[int, int],
    cin_per_group: int,
    cout_per_group: int,
    oh: int,
    ow: int,
) -> bool:
    if groups != 1:
        return False
    if kh != 1 or kw != 1 or dilation != (1, 1):
        return False

    hw = oh * ow

    # fp32 weak path:
    #   [32,256,14,14] x [512,256,1,1]
    # plus [32,128,28,28] x [256,128,1,1], which is safe for this path.
    if dtype == torch.float32:
        return cin_per_group >= 128 and cout_per_group >= 256

    # fp16/bf16 original weak path:
    #   [32,128,28,28] x [256,128,1,1]
    #
    # Avoid changing [32,256,14,14] x [512,256,1,1] for fp16/bf16 because
    # the old NCHW 1x1 kernel was already fast there.
    return (
        dtype in (torch.float16, torch.bfloat16)
        and cin_per_group == 128
        and cout_per_group >= 256
        and 512 <= hw <= 1024
    )


def _use_spatial_nchw_m_kernel(
    dtype: torch.dtype,
    groups: int,
    is_depthwise: bool,
    kh: int,
    kw: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    cin_per_group: int,
    cout_per_group: int,
    oh: int,
    ow: int,
) -> bool:
    """Use an OC x global-M NCHW implicit-GEMM kernel for weak 3x3 cases.

    The older NCHW spatial kernels tile only within one batch item.  For the
    small-spatial high-channel cases in the benchmark this creates many tiny
    GEMMs and weak L2 reuse.  This path keeps the output in NCHW order, but lets
    M = N * OH * OW span batches, so stores remain contiguous along HW while the
    tile is large enough for Tensor Core/TF32 paths.
    """
    if groups != 1 or is_depthwise:
        return False
    if kh != 3 or kw != 3 or stride != (1, 1):
        return False

    hw = oh * ow

    if dtype == torch.float32:
        return cin_per_group >= 64 and cout_per_group >= 64

    return (
        dtype in (torch.float16, torch.bfloat16)
        and dilation == (1, 1)
        and cin_per_group >= 256
        and cout_per_group >= 256
        and hw <= 14 * 14
    )



def _use_spatial_3x3_split_nchw(
    dtype: torch.dtype,
    groups: int,
    is_depthwise: bool,
    kh: int,
    kw: int,
    stride: Tuple[int, int],
    padding_2d: Tuple[int, int, int, int],
    dilation: Tuple[int, int],
    cin_per_group: int,
    cout_per_group: int,
    oh: int,
    ow: int,
) -> bool:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if groups != 1 or is_depthwise:
        return False
    if kh != 3 or kw != 3 or stride != (1, 1):
        return False
    pad_top, pad_bottom, pad_left, pad_right = padding_2d
    if pad_top != dilation[0] or pad_bottom != dilation[0]:
        return False
    if pad_left != dilation[1] or pad_right != dilation[1]:
        return False
    # Need a non-empty interior band.  The benchmark weak cases all satisfy it.
    if oh <= 2 * dilation[0] or ow <= 2 * dilation[1]:
        return False

    if dtype == torch.float32:
        return cin_per_group >= 64 and cout_per_group >= 64

    # For fp16/bf16 the split path is primarily useful for large-channel cases;
    # smaller 3x3 cases are already good with the original packed path.
    return cin_per_group >= 256 and cout_per_group >= 256


def _use_depthwise_c1_nchw(
    c_in: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> bool:
    if stride != (1, 1) or dilation != (1, 1):
        return False

    hw = oh * ow
    return (
        (c_in <= 32 and kh == 3 and kw == 3 and hw >= 112 * 112)
        or (c_in >= 64 and kh * kw >= 25 and hw <= 28 * 28)
    )


# -----------------------------------------------------------------------------
# NCHW kernels
# -----------------------------------------------------------------------------


@libentry()
@libtuner(
    configs=_CONV2D_1X1_CONFIGS,
    key=[
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW
    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW
    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_hw & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        ic_local = k0 + offs_k
        mask_k = ic_local < CIN_PER_GROUP
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        x_ptrs = (
            x_ptr
            + x_batch_base
            + ic_global[:, None] * (XH * XW)
            + ih[None, :] * XW
            + iw[None, :]
        )
        x = tl.load(x_ptrs, mask=mask_k[:, None] & valid_hw[None, :], other=0.0)

        # Packed [G, CoutG, CinG]
        w_ptrs = (
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + ic_local[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )



@libentry()
@libtuner(
    configs=_CONV2D_1X1_CONFIGS,
    key=[
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_nchw_pad0_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # Specialized NCHW 1x1, stride=1, padding=0.  It removes all oh/ow -> ih/iw
    # address arithmetic and boundary predicates from the hot path.  The weak
    # 1x1 benchmark cases are all in this form.
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    x_batch_base = batch_idx * (C_IN * HW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        ic_local = k0 + offs_k_base
        mask_k = ic_local < CIN_PER_GROUP
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        x = tl.load(
            x_ptr
            + x_batch_base
            + ic_global[:, None] * HW
            + offs_hw[None, :],
            mask=mask_k[:, None] & mask_hw[None, :],
            other=0.0,
        )

        w = tl.load(
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + ic_local[None, :],
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_1X1_NCHW_M_CONFIGS,
    key=[
        "M",
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_nchw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * (C_IN * XH * XW)
            + ic_global[None, :] * (XH * XW)
            + ih[:, None] * XW
            + iw[:, None]
        )
        x = tl.load(
            x_ptrs,
            mask=valid_hw[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Packed [G, CinG, CoutG]
        w_ptrs = (
            w_ptr
            + pid_g * (CIN_PER_GROUP * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_oc[None, :]
        )
        w = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_oc[None, :],
            other=0.0,
        )

        acc = tl.dot(x, w, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * (C_OUT * OH * OW)
        + oc_global[None, :] * (OH * OW)
        + hw[:, None]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_oc[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_1X1_NCHW_M_CONFIGS,
    key=[
        "M",
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_nchw_m_oc_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        x_ptrs = (
            x_ptr
            + batch_idx[None, :] * (C_IN * XH * XW)
            + ic_global[:, None] * (XH * XW)
            + ih[None, :] * XW
            + iw[None, :]
        )
        x = tl.load(
            x_ptrs,
            mask=mask_k[:, None] & valid_hw[None, :],
            other=0.0,
        )

        # Packed [G, CoutG, CinG]
        w_ptrs = (
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + offs_k[None, :]
        )
        w = tl.load(
            w_ptrs,
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )

        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_CONFIGS,
    key=[
        "OH",
        "OW",
        "KH",
        "KW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW
    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_hw[None, :]
            & mask_k[:, None]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = x_ptr + x_batch_base + ic_global[:, None] * (XH * XW) + ih * XW + iw
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        # Contiguous OIHW flattened as [G, CoutG, CinG*KH*KW].
        w_ptrs = (
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS,
    key=[
        "OH",
        "OW",
        "KH",
        "KW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_packed_khw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Static kh/kw loops remove div/mod by KH*KW from the hot K loop.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x_ptrs = (
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :]
                )
                x = tl.load(
                    x_ptrs,
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w_ptrs = (
                    w_ptr
                    + (
                        (
                            ((group_idx * KH + kh) * KW + kw) * COUT_PER_GROUP
                            + offs_oc[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + ic_local[None, :]
                )
                w = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )

                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_CONFIGS,
    key=[
        "M",
        "OH",
        "OW",
        "KH",
        "KW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW
    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_k[:, None]
            & mask_m[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[None, :] * (C_IN * XH * XW)
            + ic_global[:, None] * (XH * XW)
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        # Contiguous OIHW flattened as [G, CoutG, CinG*KH*KW].
        w_ptrs = (
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS,
    key=[
        "M",
        "OH",
        "OW",
        "KH",
        "KW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_m_packed_khw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Static kh/kw loops remove div/mod by KH*KW from the K loop while M spans
    # all batch items.  Accumulator orientation is OC x M so NCHW stores are
    # contiguous along the HW dimension.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_m & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = pid_g * CIN_PER_GROUP + ic_local

                x_ptrs = (
                    x_ptr
                    + batch_idx[None, :] * (C_IN * XH * XW)
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :]
                )
                x = tl.load(
                    x_ptrs,
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w_ptrs = (
                    w_ptr
                    + (
                        (
                            ((pid_g * KH + kh) * KW + kw) * COUT_PER_GROUP
                            + offs_oc[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + ic_local[None, :]
                )
                w = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )

                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )



@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS,
    key=[
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_3x3_interior_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M_INT: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # 3x3, stride=1, padding=dilation interior.  All input coordinates are
    # valid here, so the hot loop has no boundary comparisons.  Border elements
    # are produced by conv2d_spatial_nchw_3x3_border_kernel.
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    INT_H = OH - 2 * DIL_H
    INT_W = OW - 2 * DIL_W

    num_pid_m = tl.cdiv(M_INT, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_i = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_i = offs_i < M_INT
    mask_oc = offs_oc < COUT_PER_GROUP

    oh_i = offs_i // INT_W + DIL_H
    ow_i = offs_i - (oh_i - DIL_H) * INT_W + DIL_W
    out_hw = oh_i * OW + ow_i

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh_i - PAD_TOP + kh * DIL_H
        for kw in tl.static_range(0, 3):
            iw = ow_i - PAD_LEFT + kw * DIL_W
            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x = tl.load(
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :],
                    mask=mask_k[:, None] & mask_i[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w = tl.load(
                    w_ptr
                    + (((group_idx * 3 + kh) * 3 + kw) * COUT_PER_GROUP + offs_oc[:, None])
                    * CIN_PER_GROUP
                    + ic_local[None, :],
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr + y_batch_base + oc_global[:, None] * (OH * OW) + out_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_i[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS,
    key=[
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_3x3_border_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M_BORDER: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # Produces only the border band of the same 3x3/padding=dilation case.
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    num_pid_m = tl.cdiv(M_BORDER, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_b = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_b = offs_b < M_BORDER
    mask_oc = offs_oc < COUT_PER_GROUP

    TOP = DIL_H * OW
    BOTTOM = DIL_H * OW
    SIDE_W = 2 * DIL_W
    MID_H = OH - 2 * DIL_H

    in_top = offs_b < TOP
    in_bottom = (offs_b >= TOP) & (offs_b < TOP + BOTTOM)
    rem_bottom = offs_b - TOP
    rem_side = offs_b - TOP - BOTTOM

    side_row = rem_side // SIDE_W + DIL_H
    side_col_tmp = rem_side - (side_row - DIL_H) * SIDE_W
    side_col = tl.where(side_col_tmp < DIL_W, side_col_tmp, OW - DIL_W + (side_col_tmp - DIL_W))

    out_hw_top = offs_b
    out_hw_bottom = (OH - DIL_H) * OW + rem_bottom
    out_hw_side = side_row * OW + side_col
    out_hw = tl.where(in_top, out_hw_top, tl.where(in_bottom, out_hw_bottom, out_hw_side))

    oh_o = out_hw // OW
    ow_o = out_hw - oh_o * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh_o * 1 - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)
        for kw in tl.static_range(0, 3):
            iw = ow_o * 1 - PAD_LEFT + kw * DIL_W
            valid_hw = mask_b & valid_h & (iw >= 0) & (iw < XW)
            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x = tl.load(
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :],
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )
                w = tl.load(
                    w_ptr
                    + (((group_idx * 3 + kh) * 3 + kw) * COUT_PER_GROUP + offs_oc[:, None])
                    * CIN_PER_GROUP
                    + ic_local[None, :],
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr + y_batch_base + oc_global[:, None] * (OH * OW) + out_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_b[None, :],
    )


@libentry()
@libtuner(
    configs=_DW_CONV2D_V2_CONFIGS,
    key=[
        "M",
        "C_IN",
        "KH",
        "KW",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def depthwise_conv2d_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < M
    mask_c = offs_c < C_IN

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = pid_n * (C_IN * XH * XW)
    y_batch_base = pid_n * (C_IN * OH * OW)

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x_ptrs = (
                x_ptr
                + x_batch_base
                + offs_c[:, None] * (XH * XW)
                + ih[None, :] * XW
                + iw[None, :]
            )
            x = tl.load(
                x_ptrs,
                mask=mask_c[:, None] & valid_hw[None, :],
                other=0.0,
            )

            # Packed [KH, KW, C]
            w = tl.load(
                w_ptr + (kh * KW + kw) * C_IN + offs_c,
                mask=mask_c,
                other=0.0,
            )
            acc += w[:, None] * x

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + offs_c[:, None] * (OH * OW) + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_DW_CONV2D_C1_CONFIGS,
    key=[
        "M",
        "C_IN",
        "KH",
        "KW",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def depthwise_conv2d_nchw_c1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_base = x_ptr + n * (C_IN * XH * XW) + c * (XH * XW)
    y_base = y_ptr + n * (C_IN * OH * OW) + c * (OH * OW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x = tl.load(
                x_base + ih * XW + iw,
                mask=valid_hw,
                other=0.0,
            )
            ww = tl.load(w_ptr + (kh * KW + kw) * C_IN + c)
            acc += x * ww

    if HAS_BIAS:
        acc += tl.load(bias_ptr + c)

    tl.store(
        y_base + offs_hw,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_hw,
    )


# -----------------------------------------------------------------------------
# Channels-last kernels
# -----------------------------------------------------------------------------


@libentry()
@libtuner(
    configs=_CONV2D_1X1_CONFIGS,
    key=[
        "OH",
        "OW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_n = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_m = offs_m < M
    mask_n = offs_n < COUT_PER_GROUP

    HW = OH * OW
    batch_idx = offs_m // HW
    rem = offs_m - batch_idx * HW
    oh = rem // OW
    ow = rem - oh * OW
    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        a_ptrs = (
            x_ptr
            + batch_idx[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )
        a = tl.load(a_ptrs, mask=valid_hw[:, None] & mask_k[None, :], other=0.0)

        # Packed [G, CinG, CoutG]
        w_ptrs = (
            w_ptr
            + pid_g * (CIN_PER_GROUP * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_n, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_CONFIGS,
    key=[
        "OH",
        "OW",
        "KH",
        "KW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_n = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < COUT_PER_GROUP

    HW = OH * OW
    batch_idx = offs_m // HW
    rem = offs_m - batch_idx * HW
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        ih = oh[:, None] * STRIDE_H - PAD_TOP + kh_idx[None, :] * DIL_H
        iw = ow[:, None] * STRIDE_W - PAD_LEFT + kw_idx[None, :] * DIL_W
        valid = (
            mask_m[:, None]
            & mask_k[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + ih * x_stride_h
            + iw * x_stride_w
        )
        a = tl.load(x_ptrs, mask=valid, other=0.0)

        # Packed [G, CinG, KH, KW, CoutG].
        w_ptrs = (
            w_ptr
            + pid_g * (KDIM * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_n, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@libentry()
@libtuner(
    configs=_DW_CONV2D_V2_CONFIGS,
    key=[
        "M",
        "C_IN",
        "KH",
        "KW",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def depthwise_conv2d_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    C_IN,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < M
    mask_c = offs_c < C_IN

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    x_base = x_ptr + pid_n * x_stride_n
    y_base = y_ptr + pid_n * y_stride_n

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x_ptrs = (
                x_base
                + ih[:, None] * x_stride_h
                + iw[:, None] * x_stride_w
                + offs_c[None, :] * x_stride_c
            )
            x = tl.load(
                x_ptrs,
                mask=valid_hw[:, None] & mask_c[None, :],
                other=0.0,
            )

            # Packed [KH, KW, C]
            w = tl.load(
                w_ptr + (kh * KW + kw) * C_IN + offs_c,
                mask=mask_c,
                other=0.0,
            )
            acc += x * w[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_base
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
        + offs_c[None, :] * y_stride_c
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_hw[:, None] & mask_c[None, :],
    )



@triton.jit
def conv2d_fp64_im2col_nchw_kernel(
    x_ptr,
    col_ptr,
    TOTAL: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # Materialize columns as [G, N, K, HW] contiguous.  Each [K, HW] slice is a
    # column-major matrix of shape HW x K, exactly what cuBLAS DGEMM consumes.
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < TOTAL

    HW = OH * OW
    KERNEL_AREA = KH * KW
    KDIM = CIN_PER_GROUP * KERNEL_AREA

    hw = offs % HW
    k = (offs // HW) % KDIM
    bn = offs // (KDIM * HW)
    group_idx = bn // N
    batch_idx = bn - group_idx * N

    ow = hw % OW
    oh = hw // OW

    ic_local = k // KERNEL_AREA
    rem = k - ic_local * KERNEL_AREA
    kh_idx = rem // KW
    kw_idx = rem - kh_idx * KW
    ic_global = group_idx * CIN_PER_GROUP + ic_local

    ih = oh * STRIDE_H - PAD_TOP + kh_idx * DIL_H
    iw = ow * STRIDE_W - PAD_LEFT + kw_idx * DIL_W
    valid = mask & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    x = tl.load(
        x_ptr
        + batch_idx * (C_IN * XH * XW)
        + ic_global * (XH * XW)
        + ih * XW
        + iw,
        mask=valid,
        other=0.0,
    )
    tl.store(col_ptr + offs, x, mask=mask)


@triton.jit
def conv2d_add_bias_nchw_kernel(
    y_ptr,
    bias_ptr,
    TOTAL: tl.constexpr,
    HW: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < TOTAL
    oc = (offs // HW) % C_OUT
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    tl.store(y_ptr + offs, y + b, mask=mask)


@triton.jit
def depthwise_conv2d_fp64_nchw_c1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_base = x_ptr + n * (C_IN * XH * XW) + c * (XH * XW)
    y_base = y_ptr + n * (C_IN * OH * OW) + c * (OH * OW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float64)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x = tl.load(x_base + ih * XW + iw, mask=valid_hw, other=0.0).to(tl.float64)
            ww = tl.load(w_ptr + (kh * KW + kw) * C_IN + c).to(tl.float64)
            acc += x * ww

    if HAS_BIAS:
        acc += tl.load(bias_ptr + c).to(tl.float64)

    tl.store(y_base + offs_hw, acc, mask=mask_hw)


@triton.jit
def conv2d_fp64_nchw_m_tile_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float64)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0).to(tl.float64)
        acc += bias[:, None]

    # FP64 cannot rely on Tensor Core tl.dot in a portable way here.  Instead we
    # compute an OC x M tile and reduce K through small static chunks, reusing
    # each loaded input vector across BLOCK_OC output channels and each loaded
    # weight vector across BLOCK_HW spatial/batch positions.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_m & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                for kk in tl.static_range(0, BLOCK_K):
                    ic_local = k0 + kk
                    mask_k = ic_local < CIN_PER_GROUP
                    ic_global = pid_g * CIN_PER_GROUP + ic_local

                    x = tl.load(
                        x_ptr
                        + batch_idx * (C_IN * XH * XW)
                        + ic_global * (XH * XW)
                        + ih * XW
                        + iw,
                        mask=valid_hw & mask_k,
                        other=0.0,
                    ).to(tl.float64)

                    w = tl.load(
                        w_ptr
                        + oc_global * (CIN_PER_GROUP * KH * KW)
                        + (ic_local * KH + kh) * KW
                        + kw,
                        mask=mask_oc & mask_k,
                        other=0.0,
                    ).to(tl.float64)

                    acc += w[:, None] * x[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(y_ptrs, acc, mask=mask_oc[:, None] & mask_m[None, :])


# -----------------------------------------------------------------------------
# FP64 kernel
# -----------------------------------------------------------------------------
#
# This replaces the old fp64 scalar kernel.  The old version used:
#
#   for ci in tl.static_range(0, CIN_PER_GROUP)
#
# which fully unrolled large channel counts and could make Triton JIT appear to
# hang.  This version uses dynamic K-blocks.  It is not meant to beat cuDNN; it
# is meant to be correct and not hang while staying purely Triton.
# -----------------------------------------------------------------------------


@triton.jit
def conv2d_fp64_vector_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    KDIM: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_e = offs_e < total_elements

    ow = offs_e % OW
    oh = (offs_e // OW) % OH
    oc = (offs_e // (OH * OW)) % C_OUT
    batch = offs_e // (C_OUT * OH * OW)

    group = oc // COUT_PER_GROUP
    kernel_area = KH * KW

    acc = tl.zeros((BLOCK_E,), dtype=tl.float64)

    if HAS_BIAS:
        b = tl.load(bias_ptr + oc, mask=mask_e, other=0.0).to(tl.float64)
        acc += b

    offs_k_base = tl.arange(0, BLOCK_K)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // kernel_area
        rem = offs_k - ic_local * kernel_area
        kh_idx = rem // KW
        kw_idx = rem - kh_idx * KW

        ic_global = group[None, :] * CIN_PER_GROUP + ic_local[:, None]

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W

        valid = (
            mask_k[:, None]
            & mask_e[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch[None, :] * (C_IN * XH * XW)
            + ic_global * (XH * XW)
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float64)

        w_ptrs = (
            w_ptr
            + oc[None, :] * (CIN_PER_GROUP * KH * KW)
            + offs_k[:, None]
        )
        ww = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_e[None, :],
            other=0.0,
        ).to(tl.float64)

        acc += tl.sum(x * ww, axis=0)

    y_ptrs = (
        y_ptr
        + batch * (C_OUT * OH * OW)
        + oc * (OH * OW)
        + oh * OW
        + ow
    )
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_e)




def _launch_add_bias_nchw(output: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
    if bias is None:
        return
    total = output.numel()
    hw = output.shape[2] * output.shape[3]
    block_e = 256
    conv2d_add_bias_nchw_kernel[(triton.cdiv(total, block_e),)](
        output,
        bias,
        total,
        hw,
        output.shape[1],
        BLOCK_E=block_e,
    )


def _conv2d_fp64_cublas_path(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding_2d: Tuple[int, int, int, int],
    dilation: Tuple[int, int],
    groups: int,
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    cin_per_group: int,
    cout_per_group: int,
) -> Optional[torch.Tensor]:
    if not _USE_CUBLAS_FP64:
        return None
    if input.dtype != torch.float64 or not input.is_cuda:
        return None
    if groups >= 4:
        # The custom Triton grouped/depthwise paths are already strong here and
        # avoid a large im2col buffer.
        return None

    pad_top, _pad_bottom, pad_left, _pad_right = padding_2d
    hw = oh * ow
    elem_size = 8

    output = torch.empty((n, c_out, oh, ow), device=input.device, dtype=input.dtype)

    # 1x1 stride=1 pad=0 is a true batched DGEMM with no im2col.  NCHW memory is
    # interpreted as column-major matrices: X is HW x CinG, W is CinG x CoutG,
    # and Y is HW x CoutG for each batch/group.
    if kh == 1 and kw == 1 and stride == (1, 1) and padding_2d == (0, 0, 0, 0) and dilation == (1, 1):
        for g in range(groups):
            ok = _cublas_dgemm_strided_batched(
                input,
                input.data_ptr() + g * cin_per_group * hw * elem_size,
                weight.data_ptr() + g * cout_per_group * cin_per_group * elem_size,
                output.data_ptr() + g * cout_per_group * hw * elem_size,
                hw,
                cout_per_group,
                cin_per_group,
                hw,
                cin_per_group,
                hw,
                c_in * hw,
                0,
                c_out * hw,
                n,
                0.0,
            )
            if not ok:
                return None
        _launch_add_bias_nchw(output, bias)
        return output

    # General fp64 spatial path: Triton im2col + cuBLAS DGEMM.  The column buffer
    # is laid out group-major: [G, N, K, HW], where each [K, HW] matrix is
    # column-major HW x K.  This restores GEMM-level reuse without torch.mm.
    kdim = cin_per_group * kh * kw
    try:
        cols = torch.empty(
            (groups, n, kdim, hw),
            device=input.device,
            dtype=input.dtype,
        )
    except RuntimeError:
        return None

    total_cols = groups * n * kdim * hw
    block_e = 256
    conv2d_fp64_im2col_nchw_kernel[(triton.cdiv(total_cols, block_e),)](
        input,
        cols,
        total_cols,
        h,
        w,
        oh,
        ow,
        c_in,
        cin_per_group,
        groups,
        stride[0],
        stride[1],
        pad_top,
        pad_left,
        dilation[0],
        dilation[1],
        kh,
        kw,
        n,
        BLOCK_E=block_e,
    )

    for g in range(groups):
        ok = _cublas_dgemm_strided_batched(
            input,
            cols.data_ptr() + g * n * kdim * hw * elem_size,
            weight.data_ptr() + g * cout_per_group * kdim * elem_size,
            output.data_ptr() + g * cout_per_group * hw * elem_size,
            hw,
            cout_per_group,
            kdim,
            hw,
            kdim,
            hw,
            kdim * hw,
            0,
            c_out * hw,
            n,
            0.0,
        )
        if not ok:
            return None

    _launch_add_bias_nchw(output, bias)
    return output


# -----------------------------------------------------------------------------
# Public op
# -----------------------------------------------------------------------------


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    stride = _pair(stride)
    dilation = _pair(dilation)

    padding_2d = _normalize_padding(weight, stride, padding, dilation)
    _check_conv2d_inputs(input, weight, bias, stride, padding_2d, dilation, groups)

    if not input.is_cuda:
        raise NotImplementedError(
            "flag_dnn conv2d Triton implementation requires CUDA input"
        )

    n, c_in, h, w = input.shape
    c_out, c_per_group, kh, kw = weight.shape
    pad_top, pad_bottom, pad_left, pad_right = padding_2d

    oh = _conv_out_dim(h, pad_top, pad_bottom, dilation[0], kh, stride[0])
    ow = _conv_out_dim(w, pad_left, pad_right, dilation[1], kw, stride[1])

    if oh < 0 or ow < 0:
        raise RuntimeError("computed output size is negative")
    if oh == 0 or ow == 0:
        return torch.empty(
            (n, c_out, max(oh, 0), max(ow, 0)),
            device=input.device,
            dtype=input.dtype,
        )

    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    cout_per_group = c_out // groups
    cin_per_group = c_in // groups

    is_depthwise = groups == c_in and c_per_group == 1 and c_out == c_in
    is_1x1 = kh == 1 and kw == 1 and dilation == (1, 1)
    dtype_id = _dtype_id(input.dtype)

    # FP64 path.  Prefer Triton im2col + low-level cuBLAS DGEMM when available:
    # this avoids torch.mm / torch.matmul while restoring GEMM-level fp64 reuse.
    # If cuBLAS is unavailable, fall back to the pure Triton tiled direct kernel.
    if input.dtype == torch.float64:
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        with torch_device_fn.device(input.device):
            cublas_out = _conv2d_fp64_cublas_path(
                input,
                weight,
                bias,
                stride,
                padding_2d,
                dilation,
                groups,
                n,
                c_in,
                h,
                w,
                c_out,
                kh,
                kw,
                oh,
                ow,
                cin_per_group,
                cout_per_group,
            )
            if cublas_out is not None:
                return cublas_out

        output = torch.empty((n, c_out, oh, ow), device=input.device, dtype=input.dtype)

        with torch_device_fn.device(input.device):
            if is_depthwise:
                w_dw = _pack_depthwise_weight_khw_c(weight, groups)
                block_hw = 256 if kh * kw <= 9 else 128

                depthwise_conv2d_fp64_nchw_c1_kernel[
                    (triton.cdiv(oh * ow, block_hw), c_in, n)
                ](
                    input,
                    w_dw,
                    bias if bias is not None else output,
                    output,
                    oh * ow,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                    BLOCK_HW=block_hw,
                )
                return output

            m = n * oh * ow

            # Keep fp64 tiles modest to control register pressure.  1x1 and
            # group-conv have smaller K and can use a slightly wider OC tile;
            # dense 3x3 high-channel cases use a narrower OC tile.
            if is_1x1 or groups > 1:
                block_oc = 16
                block_hw = 16
                block_k = 8
            else:
                block_oc = 8 if cin_per_group >= 256 else 16
                block_hw = 16
                block_k = 8

            grid_fp64 = (
                triton.cdiv(m, block_hw) * triton.cdiv(cout_per_group, block_oc),
                groups,
            )

            conv2d_fp64_nchw_m_tile_kernel[grid_fp64](
                input,
                weight,
                bias if bias is not None else output,
                output,
                m,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                BLOCK_OC=block_oc,
                BLOCK_HW=block_hw,
                BLOCK_K=block_k,
                GROUP_M=_GROUP_SIZE_M,
                num_warps=4,
                num_stages=1,
            )

        return output

    use_channels_last = _input_has_fast_channel_stride(input)

    with torch_device_fn.device(input.device):
        if use_channels_last:
            output = torch.empty(
                (n, c_out, oh, ow),
                device=input.device,
                dtype=input.dtype,
                memory_format=torch.channels_last,
            )

            if is_depthwise:
                w_dw = _pack_depthwise_weight_khw_c(weight, groups)

                def grid_dw(meta):
                    return (
                        triton.cdiv(oh * ow, meta["BLOCK_HW"]),
                        triton.cdiv(c_in, meta["BLOCK_C"]),
                        n,
                    )

                depthwise_conv2d_cl_kernel[grid_dw](
                    input,
                    w_dw,
                    bias if bias is not None else output,
                    output,
                    oh * ow,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    input.stride(0),
                    input.stride(1),
                    input.stride(2),
                    input.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                    DTYPE_ID=dtype_id,
                )
                return output

            if is_1x1:
                w_1x1 = _pack_weight_1x1_cl(weight, groups)
                m = n * oh * ow

                def grid_1x1_cl(meta):
                    return (
                        triton.cdiv(m, meta["BLOCK_HW"])
                        * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                        groups,
                    )

                conv2d_1x1_cl_kernel[grid_1x1_cl](
                    input,
                    w_1x1,
                    bias if bias is not None else output,
                    output,
                    m,
                    h,
                    w,
                    oh,
                    ow,
                    input.stride(0),
                    input.stride(1),
                    input.stride(2),
                    input.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    cin_per_group,
                    cout_per_group,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    HAS_BIAS=bias is not None,
                    GROUP_M=_GROUP_SIZE_M,
                    DTYPE_ID=dtype_id,
                )
                return output

            w_spatial = _pack_weight_spatial_cl(weight, groups)
            m = n * oh * ow

            def grid_spatial_cl(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    groups,
                )

            conv2d_spatial_cl_kernel[grid_spatial_cl](
                input,
                w_spatial,
                bias if bias is not None else output,
                output,
                m,
                h,
                w,
                oh,
                ow,
                input.stride(0),
                input.stride(1),
                input.stride(2),
                input.stride(3),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                cin_per_group,
                cout_per_group,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
                DTYPE_ID=dtype_id,
            )
            return output

        # NCHW/default path.
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = torch.empty((n, c_out, oh, ow), device=input.device, dtype=input.dtype)

        if is_depthwise:
            w_dw = _pack_depthwise_weight_khw_c(weight, groups)

            if _use_depthwise_c1_nchw(c_in, kh, kw, oh, ow, stride, dilation):

                def grid_dw_c1(meta):
                    return (
                        triton.cdiv(oh * ow, meta["BLOCK_HW"]),
                        c_in,
                        n,
                    )

                depthwise_conv2d_nchw_c1_kernel[grid_dw_c1](
                    input,
                    w_dw,
                    bias if bias is not None else output,
                    output,
                    oh * ow,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                    DTYPE_ID=dtype_id,
                )
                return output

            def grid_dw_nchw(meta):
                return (
                    triton.cdiv(oh * ow, meta["BLOCK_HW"]),
                    triton.cdiv(c_in, meta["BLOCK_C"]),
                    n,
                )

            depthwise_conv2d_nchw_kernel[grid_dw_nchw](
                input,
                w_dw,
                bias if bias is not None else output,
                output,
                oh * ow,
                h,
                w,
                oh,
                ow,
                c_in,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                DTYPE_ID=dtype_id,
            )
            return output

        if is_1x1:
            if stride == (1, 1) and padding_2d == (0, 0, 0, 0):
                w_1x1_fast = _pack_weight_1x1_nchw(weight, groups)

                def grid_1x1_fast(meta):
                    return (
                        triton.cdiv(oh * ow, meta["BLOCK_HW"])
                        * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                        n * groups,
                    )

                conv2d_1x1_nchw_pad0_kernel[grid_1x1_fast](
                    input,
                    w_1x1_fast,
                    bias if bias is not None else output,
                    output,
                    oh * ow,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    groups,
                    HAS_BIAS=bias is not None,
                    GROUP_M=_GROUP_SIZE_M,
                    DTYPE_ID=dtype_id,
                )
                return output

            if _use_1x1_nchw_m_kernel(
                input.dtype,
                groups,
                kh,
                kw,
                dilation,
                cin_per_group,
                cout_per_group,
                oh,
                ow,
            ):
                w_1x1_m = _pack_weight_1x1_nchw(weight, groups)
                m = n * oh * ow

                def grid_1x1_nchw_m_oc(meta):
                    return (
                        triton.cdiv(m, meta["BLOCK_HW"])
                        * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                        groups,
                    )

                conv2d_1x1_nchw_m_oc_kernel[grid_1x1_nchw_m_oc](
                    input,
                    w_1x1_m,
                    bias if bias is not None else output,
                    output,
                    m,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    groups,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    HAS_BIAS=bias is not None,
                    GROUP_M=_GROUP_SIZE_M,
                    DTYPE_ID=dtype_id,
                )
                return output

            w_1x1 = _pack_weight_1x1_nchw(weight, groups)

            def grid_1x1_nchw(meta):
                return (
                    triton.cdiv(oh * ow, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    n * groups,
                )

            conv2d_1x1_nchw_kernel[grid_1x1_nchw](
                input,
                w_1x1,
                bias if bias is not None else output,
                output,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
                DTYPE_ID=dtype_id,
            )
            return output

        def grid_spatial_nchw(meta):
            return (
                triton.cdiv(oh * ow, meta["BLOCK_HW"])
                * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                n * groups,
            )

        if _use_spatial_3x3_split_nchw(
            input.dtype,
            groups,
            is_depthwise,
            kh,
            kw,
            stride,
            padding_2d,
            dilation,
            cin_per_group,
            cout_per_group,
            oh,
            ow,
        ):
            w_spatial_split = _pack_weight_spatial_nchw_khw_oci(weight, groups)
            m_int = (oh - 2 * dilation[0]) * (ow - 2 * dilation[1])
            m_border = oh * ow - m_int

            def grid_spatial_split_int(meta):
                return (
                    triton.cdiv(m_int, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    n * groups,
                )

            conv2d_spatial_nchw_3x3_interior_kernel[grid_spatial_split_int](
                input,
                w_spatial_split,
                bias if bias is not None else output,
                output,
                m_int,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
                DTYPE_ID=dtype_id,
            )

            def grid_spatial_split_border(meta):
                return (
                    triton.cdiv(m_border, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    n * groups,
                )

            conv2d_spatial_nchw_3x3_border_kernel[grid_spatial_split_border](
                input,
                w_spatial_split,
                bias if bias is not None else output,
                output,
                m_border,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
                DTYPE_ID=dtype_id,
            )
            return output

        if _use_spatial_nchw_m_kernel(
            input.dtype,
            groups,
            is_depthwise,
            kh,
            kw,
            stride,
            dilation,
            cin_per_group,
            cout_per_group,
            oh,
            ow,
        ):
            m = n * oh * ow

            def grid_spatial_nchw_m(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    groups,
                )

            if input.dtype == torch.float32:
                conv2d_spatial_nchw_m_kernel[grid_spatial_nchw_m](
                    input,
                    weight,
                    bias if bias is not None else output,
                    output,
                    m,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    groups,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                    GROUP_M=_GROUP_SIZE_M,
                    DTYPE_ID=dtype_id,
                )
            else:
                w_spatial_nchw_m = _pack_weight_spatial_nchw_khw_oci(weight, groups)
                conv2d_spatial_nchw_m_packed_khw_kernel[grid_spatial_nchw_m](
                    input,
                    w_spatial_nchw_m,
                    bias if bias is not None else output,
                    output,
                    m,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    groups,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                    GROUP_M=_GROUP_SIZE_M,
                    DTYPE_ID=dtype_id,
                )
            return output

        if _use_packed_spatial_nchw(
            input.dtype,
            groups,
            kh,
            kw,
            stride,
            dilation,
            cin_per_group,
            cout_per_group,
            oh,
            ow,
        ):
            w_spatial_nchw = _pack_weight_spatial_nchw_khw_oci(weight, groups)

            conv2d_spatial_nchw_packed_khw_kernel[grid_spatial_nchw](
                input,
                w_spatial_nchw,
                bias if bias is not None else output,
                output,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
                DTYPE_ID=dtype_id,
            )
        else:
            conv2d_spatial_nchw_kernel[grid_spatial_nchw](
                input,
                weight,
                bias if bias is not None else output,
                output,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
                DTYPE_ID=dtype_id,
            )

        return output