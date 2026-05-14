import logging
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_INDEX_DTYPES = (torch.int32, torch.int64)


def _is_index_tensor(arg: Any) -> bool:
    return torch.is_tensor(arg) and arg.dtype in _INDEX_DTYPES


def _arg_or_kw(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    pos: int,
    name: str,
    default: Any,
) -> Any:
    if len(args) > pos:
        if name in kwargs:
            raise TypeError(f"embedding got multiple values for {name}")
        return args[pos]
    return kwargs.pop(name, default)


def _canonical_padding_idx(padding_idx: Any, num_embeddings: int) -> int:
    if padding_idx is None:
        return -1

    padding_idx = int(padding_idx)
    if padding_idx > 0:
        assert (
            padding_idx < num_embeddings
        ), "Padding_idx must be within num_embeddings"
    elif padding_idx < 0:
        assert (
            padding_idx >= -num_embeddings
        ), "Padding_idx must be within num_embeddings"
        padding_idx = num_embeddings + padding_idx
    return padding_idx


def _parse_embedding_args(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, int, bool, bool]:
    if len(args) < 2:
        raise TypeError("embedding expected at least 2 positional arguments")

    kwargs = dict(kwargs)

    if _is_index_tensor(args[0]):
        if len(args) > 7:
            raise TypeError("embedding got too many positional arguments")

        indices = args[0]
        weight = args[1]
        padding_idx = _arg_or_kw(args, kwargs, 2, "padding_idx", None)
        max_norm = _arg_or_kw(args, kwargs, 3, "max_norm", None)
        _ = _arg_or_kw(args, kwargs, 4, "norm_type", 2.0)
        scale_grad_by_freq = _arg_or_kw(
            args, kwargs, 5, "scale_grad_by_freq", False
        )
        sparse = _arg_or_kw(args, kwargs, 6, "sparse", False)

        if max_norm is not None:
            raise NotImplementedError(
                "flag_dnn embedding does not support max_norm yet"
            )
    else:
        if len(args) > 5:
            raise TypeError("aten embedding got too many positional arguments")

        weight = args[0]
        indices = args[1]
        padding_idx = _arg_or_kw(args, kwargs, 2, "padding_idx", -1)
        scale_grad_by_freq = _arg_or_kw(
            args, kwargs, 3, "scale_grad_by_freq", False
        )
        sparse = _arg_or_kw(args, kwargs, 4, "sparse", False)

        max_norm = kwargs.pop("max_norm", None)
        if max_norm is not None:
            raise NotImplementedError(
                "flag_dnn embedding does not support max_norm yet"
            )
        kwargs.pop("norm_type", None)

    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"embedding got an unexpected keyword '{unexpected}'")

    if not torch.is_tensor(weight):
        raise TypeError("embedding weight must be a Tensor")
    if not torch.is_tensor(indices):
        raise TypeError("embedding input must be a Tensor")

    padding_idx = _canonical_padding_idx(padding_idx, weight.shape[0])
    return indices, weight, padding_idx, bool(scale_grad_by_freq), bool(sparse)


def _choose_block_d(embedding_dim: int) -> int:
    if embedding_dim <= 0:
        return 1
    if embedding_dim > 512:
        return min(1024, triton.next_power_of_2(embedding_dim))
    if embedding_dim > 128:
        return min(256, triton.next_power_of_2(embedding_dim))
    return max(16, triton.next_power_of_2(embedding_dim))


def _choose_block_m(embedding_dim: int) -> int:
    if embedding_dim <= 16:
        return 16
    if embedding_dim <= 32:
        return 8
    if embedding_dim <= 128:
        return 4
    if embedding_dim <= 256:
        return 2
    return 1


def _choose_num_warps(block_m: int, block_d: int) -> int:
    if block_d >= 1024:
        return 4
    tile_elems = block_m * block_d
    if tile_elems <= 64:
        return 1
    if tile_elems <= 256:
        return 4
    return 8


@triton.jit
def _embedding_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    n_indices,
    vocab_size,
    embedding_dim,
    stride_w0,
    stride_w1,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_d = tle.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < n_indices
    mask_d = offs_d < embedding_dim

    rows = tl.load(indices_ptr + offs_m, mask=mask_m, other=0)
    valid_rows = (rows >= 0) & (rows < vocab_size)
    tl.device_assert(valid_rows | ~mask_m, "embedding index out of range")

    values = tl.load(
        weight_ptr + rows[:, None] * stride_w0 + offs_d[None, :] * stride_w1,
        mask=mask_m[:, None] & mask_d[None, :] & valid_rows[:, None],
        other=0.0,
    )

    tl.store(
        out_ptr + offs_m[:, None] * embedding_dim + offs_d[None, :],
        values,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def embedding(*args, **kwargs) -> torch.Tensor:
    logger.debug("FLAG_DNN EMBEDDING")

    indices, weight, _, _, _ = _parse_embedding_args(args, kwargs)

    if indices.dtype not in _INDEX_DTYPES:
        raise RuntimeError(
            "Expected tensor for argument #1 'indices' to have scalar type "
            f"Long or Int, but got {indices.dtype}"
        )
    if indices.device != weight.device:
        raise RuntimeError(
            "embedding input and weight must be on the same device, "
            f"but got {indices.device} and {weight.device}"
        )
    if weight.device.type != runtime.device.name:
        raise NotImplementedError(
            f"flag_dnn embedding only supports {runtime.device.name} tensors"
        )
    if weight.dim() != 2:
        raise RuntimeError(
            f"'weight' must be 2-D, but got {weight.dim()}-D tensor"
        )
    if not weight.dtype.is_floating_point:
        raise NotImplementedError(
            f"flag_dnn embedding does not support weight dtype={weight.dtype}"
        )
    if weight.dtype == torch.float64 and not runtime.device.support_fp64:
        raise RuntimeError("Device does not support float64")
    if weight.requires_grad and torch.is_grad_enabled():
        raise NotImplementedError(
            "flag_dnn embedding currently supports forward inference only; "
            "backward, scale_grad_by_freq and sparse gradients are not "
            "implemented"
        )

    if not indices.is_contiguous():
        indices = indices.contiguous()

    vocab_size = weight.shape[0]
    embedding_dim = weight.shape[1]
    output_shape = tuple(indices.shape) + (embedding_dim,)
    out = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)

    n_indices = indices.numel()
    if n_indices == 0 or embedding_dim == 0:
        return out
    if vocab_size == 0:
        raise RuntimeError("index out of range in embedding")

    block_d = _choose_block_d(embedding_dim)
    block_m = _choose_block_m(embedding_dim)
    num_warps = _choose_num_warps(block_m, block_d)
    grid = (
        triton.cdiv(n_indices, block_m),
        triton.cdiv(embedding_dim, block_d),
    )

    with torch_device_fn.device(weight.device):
        _embedding_kernel[grid](
            indices,
            weight,
            out,
            n_indices,
            vocab_size,
            embedding_dim,
            weight.stride(0),
            weight.stride(1),
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=3,
        )

    return out


def embedding_renorm_(*args, **kwargs):
    raise NotImplementedError(
        "flag_dnn embedding does not support max_norm/embedding_renorm_ yet"
    )
