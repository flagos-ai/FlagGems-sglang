import logging
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _fallback_dot_configs() -> Sequence[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_SIZE": 128, "UNROLL": 2}, num_warps=2, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "UNROLL": 4}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 256, "UNROLL": 2}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 256, "UNROLL": 4}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE": 512, "UNROLL": 4}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE": 512, "UNROLL": 8}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE": 1024, "UNROLL": 4}, num_warps=8, num_stages=2
        ),
    ]


def _fallback_dot_fp64_configs() -> Sequence[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_SIZE": 64, "UNROLL": 2}, num_warps=2, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "UNROLL": 2}, num_warps=2, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "UNROLL": 4}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 256, "UNROLL": 2}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE": 256, "UNROLL": 4}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE": 512, "UNROLL": 2}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE": 512, "UNROLL": 4}, num_warps=8, num_stages=2
        ),
    ]


def _get_tuned_or_default(name: str, fallback):
    try:
        cfg = runtime.get_tuned_config(name)
        if cfg:
            return cfg
    except Exception:
        pass
    return fallback


_DOT_CONFIGS = _get_tuned_or_default("dot", _fallback_dot_configs())
_DOT_FP64_CONFIGS = _get_tuned_or_default(
    "dot_fp64", _fallback_dot_fp64_configs()
)

# single-program 阈值
_DOT_SINGLE_THRESHOLD_LOW_PREC = 24576
_DOT_SINGLE_THRESHOLD_FP32 = 16384
_DOT_SINGLE_THRESHOLD_FP64 = 2048

# 低精度 fallback 区间
_DOT_FP16_FALLBACK_MAX = 131072
_DOT_BF16_FALLBACK_MAX = 2097152


@triton.jit
def dot_single_kernel_fp32(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for tile_id in tl.static_range(0, MAX_TILES):
        tile_base = tile_id * BLOCK_SIZE * UNROLL
        for inner in tl.static_range(0, UNROLL):
            idx = tile_base + inner * BLOCK_SIZE + offs
            mask = idx < n_elements

            x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            acc += x * y

    tl.store(out_ptr, tl.sum(acc, axis=0))


@triton.jit
def dot_single_kernel_fp64(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float64)

    for tile_id in tl.static_range(0, MAX_TILES):
        tile_base = tile_id * BLOCK_SIZE * UNROLL
        for inner in tl.static_range(0, UNROLL):
            idx = tile_base + inner * BLOCK_SIZE + offs
            mask = idx < n_elements

            x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float64)
            y = tl.load(y_ptr + idx, mask=mask, other=0.0).to(tl.float64)
            acc += x * y

    tl.store(out_ptr, tl.sum(acc, axis=0))


@libentry()
@libtuner(
    configs=_DOT_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def dot_atomic_kernel_fp32(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tle.program_id(0)
    tile_size = BLOCK_SIZE * UNROLL
    block_start = pid * tile_size
    block_start = tl.multiple_of(block_start, BLOCK_SIZE)

    offs = block_start + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for inner in tl.static_range(0, UNROLL):
        idx = offs + inner * BLOCK_SIZE
        mask = idx < n_elements

        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc += x * y

    block_sum = tl.sum(acc, axis=0)
    tl.atomic_add(out_ptr, block_sum)


@libentry()
@libtuner(
    configs=_DOT_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def dot_atomic_kernel_fp64(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tle.program_id(0)
    tile_size = BLOCK_SIZE * UNROLL
    block_start = pid * tile_size
    block_start = tl.multiple_of(block_start, BLOCK_SIZE)

    offs = block_start + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float64)

    for inner in tl.static_range(0, UNROLL):
        idx = offs + inner * BLOCK_SIZE
        mask = idx < n_elements

        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float64)
        y = tl.load(y_ptr + idx, mask=mask, other=0.0).to(tl.float64)
        acc += x * y

    block_sum = tl.sum(acc, axis=0)
    tl.atomic_add(out_ptr, block_sum)


@triton.jit
def cast_scalar_kernel(src_ptr, dst_ptr):
    x = tl.load(src_ptr)
    tl.store(dst_ptr, x)


def _check_dot_inputs(input: torch.Tensor, tensor: torch.Tensor) -> None:
    if input.dim() != 1 or tensor.dim() != 1:
        raise RuntimeError("flag_dnn dot expects both input tensors to be 1D")

    if input.numel() != tensor.numel():
        raise RuntimeError(
            "inconsistent tensor size, expected both vectors to have the "
            "same number of elements"
        )

    if input.device != tensor.device:
        raise RuntimeError(
            f"flag_dnn dot device mismatch: {input.device} vs {tensor.device}"
        )

    if input.dtype != tensor.dtype:
        raise RuntimeError(
            f"flag_dnn dot dtype mismatch: {input.dtype} vs {tensor.dtype}"
        )

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn dot does not support dtype={input.dtype}"
        )


def _prepare_out(
    out: Optional[torch.Tensor],
    ref: torch.Tensor,
) -> torch.Tensor:
    if out is None:
        return torch.empty((), device=ref.device, dtype=ref.dtype)

    if out.device != ref.device:
        raise RuntimeError(
            f"dot out tensor device mismatch: expected {ref.device}, "
            f"got {out.device}"
        )

    if out.dtype != ref.dtype:
        raise RuntimeError(
            f"dot out tensor dtype mismatch: expected {ref.dtype}, "
            f"got {out.dtype}"
        )

    out.resize_(())
    return out


def _launch_atomic_fp32(
    input: torch.Tensor,
    tensor: torch.Tensor,
    n_elements: int,
    out_tensor: torch.Tensor,
) -> torch.Tensor:
    def grid(meta):
        return (
            triton.cdiv(
                n_elements,
                meta["BLOCK_SIZE"] * meta["UNROLL"],
            ),
        )

    if input.dtype == torch.float32:
        out_tensor.zero_()
        dot_atomic_kernel_fp32[grid](
            input,
            tensor,
            out_tensor,
            n_elements,
        )
        return out_tensor

    scratch = torch.zeros((), device=input.device, dtype=torch.float32)
    dot_atomic_kernel_fp32[grid](
        input,
        tensor,
        scratch,
        n_elements,
    )
    cast_scalar_kernel[(1,)](scratch, out_tensor)
    return out_tensor


def dot(
    input: torch.Tensor,
    tensor: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN DOT")

    _check_dot_inputs(input, tensor)

    if not input.is_contiguous():
        input = input.contiguous()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    n_elements = input.numel()
    out_tensor = _prepare_out(out, input)

    if n_elements == 0:
        out_tensor.zero_()
        return out_tensor

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            # 512 * 4 * 1 = 2048
            if n_elements <= _DOT_SINGLE_THRESHOLD_FP64:
                dot_single_kernel_fp64[(1,)](
                    input,
                    tensor,
                    out_tensor,
                    n_elements,
                    BLOCK_SIZE=512,
                    UNROLL=4,
                    MAX_TILES=1,
                )
                return out_tensor

            out_tensor.zero_()

            def grid(meta):
                return (
                    triton.cdiv(
                        n_elements,
                        meta["BLOCK_SIZE"] * meta["UNROLL"],
                    ),
                )

            dot_atomic_kernel_fp64[grid](
                input,
                tensor,
                out_tensor,
                n_elements,
            )
            return out_tensor

        if input.dtype == torch.float32:
            # 1024 * 8 * 2 = 16384
            if n_elements <= _DOT_SINGLE_THRESHOLD_FP32:
                single_kernel = (
                    dot_single_kernel_fp64
                    if runtime.device.support_fp64
                    else dot_single_kernel_fp32
                )
                single_kernel[(1,)](
                    input,
                    tensor,
                    out_tensor,
                    n_elements,
                    BLOCK_SIZE=1024,
                    UNROLL=8,
                    MAX_TILES=2,
                )
                return out_tensor

            return _launch_atomic_fp32(
                input,
                tensor,
                n_elements,
                out_tensor,
            )

        if input.dtype == torch.float16:
            if n_elements <= _DOT_SINGLE_THRESHOLD_LOW_PREC:
                # 1024 * 8 * 3 = 24576
                dot_single_kernel_fp32[(1,)](
                    input,
                    tensor,
                    out_tensor,
                    n_elements,
                    BLOCK_SIZE=1024,
                    UNROLL=8,
                    MAX_TILES=3,
                )
                return out_tensor

            if n_elements <= _DOT_FP16_FALLBACK_MAX:
                return _launch_atomic_fp32(
                    input,
                    tensor,
                    n_elements,
                    out_tensor,
                )

            return _launch_atomic_fp32(
                input,
                tensor,
                n_elements,
                out_tensor,
            )

        # bfloat16
        if n_elements <= _DOT_SINGLE_THRESHOLD_LOW_PREC:
            dot_single_kernel_fp32[(1,)](
                input,
                tensor,
                out_tensor,
                n_elements,
                BLOCK_SIZE=1024,
                UNROLL=8,
                MAX_TILES=3,
            )
            return out_tensor

        if n_elements <= _DOT_BF16_FALLBACK_MAX:
            return _launch_atomic_fp32(
                input,
                tensor,
                n_elements,
                out_tensor,
            )

        return _launch_atomic_fp32(
            input,
            tensor,
            n_elements,
            out_tensor,
        )
