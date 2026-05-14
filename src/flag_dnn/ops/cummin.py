import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn import runtime  # noqa: F401  (kept for parity with other ops)
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _cummin_combine(lv, li, rv, ri):
    """Associative combine for cumulative minimum."""
    r_nan = rv != rv
    l_nan = lv != lv
    # Match PyTorch CUDA cummin: tie -> take right; rhs-NaN -> take right (NaN
    # wins); lhs-NaN (only) -> take right (propagate the NaN already on rhs's
    # left).
    take_right = tl.where(r_nan, True, tl.where(l_nan, False, rv <= lv))
    v = tl.where(take_right, rv, lv)
    i = tl.where(take_right, ri, li)
    return v, i


@libentry()
@triton.jit
def cummin_inner_single_kernel(
    x_ptr,
    v_ptr,
    i_ptr,
    M,
    N,
    IDENTITY,
    BLOCK_N: tl.constexpr,
):
    """One-shot scan when N <= BLOCK_N (no carry needed)."""
    pid_m = tle.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + pid_m * N + offs, mask=mask, other=IDENTITY)
    idx_local = offs.to(tl.int32)
    v, i_local = tl.associative_scan(
        (x, idx_local), axis=0, combine_fn=_cummin_combine
    )
    i = i_local.to(tl.int64)
    tl.store(v_ptr + pid_m * N + offs, v, mask=mask)
    tl.store(i_ptr + pid_m * N + offs, i, mask=mask)


@libentry()
@triton.jit
def cummin_inner_pass1_kernel(
    x_ptr,
    v_ptr,
    i_ptr,
    carry_v_ptr,
    carry_i_ptr,
    M,
    N,
    NUM_BLOCKS,
    IDENTITY,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    base_n = pid_n * BLOCK_N
    offs = base_n + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + pid_m * N + offs, mask=mask, other=IDENTITY)
    idx_local = tl.arange(0, BLOCK_N).to(tl.int32)
    v, i_local = tl.associative_scan(
        (x, idx_local), axis=0, combine_fn=_cummin_combine
    )
    i_global = i_local.to(tl.int64) + base_n
    tl.store(v_ptr + pid_m * N + offs, v, mask=mask)
    tl.store(i_ptr + pid_m * N + offs, i_global, mask=mask)

    in_block = tl.arange(0, BLOCK_N)
    block_len = tl.minimum(BLOCK_N, N - base_n)
    sel = in_block == (block_len - 1)
    last_v = tl.sum(tl.where(sel, v, 0), axis=0)
    last_i = tl.sum(tl.where(sel, i_global, tl.zeros_like(i_global)), axis=0)
    tl.store(carry_v_ptr + pid_m * NUM_BLOCKS + pid_n, last_v)
    tl.store(carry_i_ptr + pid_m * NUM_BLOCKS + pid_n, last_i)


@libentry()
@triton.jit
def cummin_inner_carry_scan_kernel(
    carry_v_ptr,
    carry_i_ptr,
    M,
    NUM_BLOCKS,
    IDENTITY,
    BLOCK: tl.constexpr,
):
    """Scan the per-row carry array (length NUM_BLOCKS) in one block."""
    pid_m = tle.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < NUM_BLOCKS
    v = tl.load(
        carry_v_ptr + pid_m * NUM_BLOCKS + offs, mask=mask, other=IDENTITY
    )
    i = tl.load(carry_i_ptr + pid_m * NUM_BLOCKS + offs, mask=mask, other=0)
    v, i = tl.associative_scan((v, i), axis=0, combine_fn=_cummin_combine)
    tl.store(carry_v_ptr + pid_m * NUM_BLOCKS + offs, v, mask=mask)
    tl.store(carry_i_ptr + pid_m * NUM_BLOCKS + offs, i, mask=mask)


@libentry()
@triton.jit
def cummin_inner_pass2_kernel(
    v_ptr,
    i_ptr,
    carry_v_ptr,
    carry_i_ptr,
    M,
    N,
    NUM_BLOCKS,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    if pid_n == 0:
        return
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    v = tl.load(v_ptr + pid_m * N + offs, mask=mask, other=0)
    i = tl.load(i_ptr + pid_m * N + offs, mask=mask, other=0)
    cv = tl.load(carry_v_ptr + pid_m * NUM_BLOCKS + pid_n - 1)
    ci = tl.load(carry_i_ptr + pid_m * NUM_BLOCKS + pid_n - 1)
    r_nan = v != v
    l_nan = cv != cv
    take_right = tl.where(r_nan, True, tl.where(l_nan, False, v <= cv))
    new_v = tl.where(take_right, v, cv)
    new_i = tl.where(take_right, i, ci)
    tl.store(v_ptr + pid_m * N + offs, new_v, mask=mask)
    tl.store(i_ptr + pid_m * N + offs, new_i, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M_POST": 128}, num_warps=4),
        triton.Config({"BLOCK_M_POST": 256}, num_warps=4),
        triton.Config({"BLOCK_M_POST": 512}, num_warps=8),
        triton.Config({"BLOCK_M_POST": 1024}, num_warps=8),
    ],
    key=["M_pre", "M_post"],
)
@libentry()
@triton.jit
def cummin_outer_1d_kernel(
    in_ptr,
    v_ptr,
    i_ptr,
    M_pre,
    N,
    M_post,
    IDENTITY,
    BLOCK_M_POST: tl.constexpr,
):
    pid_pre = tle.program_id(0)
    pid_post = tle.program_id(1)

    offs_post = pid_post * BLOCK_M_POST + tl.arange(0, BLOCK_M_POST)
    mask_post = offs_post < M_post

    base_idx = pid_pre * (N * M_post) + offs_post

    running_v = tl.full(
        [BLOCK_M_POST], IDENTITY, dtype=in_ptr.dtype.element_ty
    )
    running_i = tl.zeros([BLOCK_M_POST], dtype=tl.int64)

    for n in range(N):
        idx = base_idx + n * M_post
        val = tl.load(in_ptr + idx, mask=mask_post, other=IDENTITY)

        r_nan = val != val
        l_nan = running_v != running_v

        take_right = tl.where(
            r_nan, True, tl.where(l_nan, False, val <= running_v)
        )

        running_v = tl.where(take_right, val, running_v)
        current_n = tl.full([BLOCK_M_POST], n, dtype=tl.int64)
        running_i = tl.where(take_right, current_n, running_i)

        tl.store(v_ptr + idx, running_v, mask=mask_post)
        tl.store(i_ptr + idx, running_i, mask=mask_post)


_INT_IDENTITY = {
    torch.int8: 127,
    torch.int16: 32767,
    torch.int32: 2147483647,
    torch.int64: 9223372036854775807,
    torch.uint8: 255,
    torch.bool: 1,
}


def _identity_for(dtype: torch.dtype):
    if dtype.is_floating_point:
        return float("inf")
    return _INT_IDENTITY[dtype]


def _pick_block_n(N: int) -> int:
    if N <= 128:
        return 128
    if N <= 256:
        return 256
    if N <= 512:
        return 512
    if N <= 1024:
        return 1024
    return 2048


def cummin(
    input: torch.Tensor,
    dim: int,
    *,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    logger.debug(f"FLAG_DNN CUMMIN (dim={dim})")

    if input.ndim == 0:
        values = input.detach().clone()
        indices = torch.zeros_like(input, dtype=torch.int64)
        if out is not None:
            _resize_for_reuse(out[0], values.shape)
            _resize_for_reuse(out[1], indices.shape)
            out[0].copy_(values)
            out[1].copy_(indices)
            return torch.return_types.cummin((out[0], out[1]))
        return torch.return_types.cummin((values, indices))

    dim = dim % input.ndim
    shape = input.shape

    if input.numel() == 0:
        values = torch.empty_like(input)
        indices = torch.empty(shape, dtype=torch.int64, device=input.device)
        if out is not None:
            _resize_for_reuse(out[0], shape)
            _resize_for_reuse(out[1], shape)
            return torch.return_types.cummin((out[0], out[1]))
        return torch.return_types.cummin((values, indices))

    N = shape[dim]
    M = 1
    for s in shape[:dim]:
        M *= s
    K = 1
    for s in shape[dim + 1 :]:
        K *= s

    input_c = input.contiguous()

    if out is None:
        values = torch.empty_like(input_c)
        indices = torch.empty(shape, dtype=torch.int64, device=input.device)
        out_provided = False
    else:
        values, indices = out
        out_provided = True
        _resize_for_reuse(values, shape)
        _resize_for_reuse(indices, shape)
        if not values.is_contiguous() or values.dtype != input.dtype:
            _values_scratch = torch.empty_like(input_c)
        else:
            _values_scratch = values
        if not indices.is_contiguous() or indices.dtype != torch.int64:
            _indices_scratch = torch.empty(
                shape, dtype=torch.int64, device=input.device
            )
        else:
            _indices_scratch = indices
        values, indices = _values_scratch, _indices_scratch

    if N == 1:
        values.copy_(input_c)
        indices.zero_()
        if out_provided:
            return _finalize_out(out, values, indices)
        return torch.return_types.cummin((values, indices))

    identity = _identity_for(input.dtype)

    with torch_device_fn.device(input.device):
        if K == 1:
            _run_inner(input_c, values, indices, M, N, identity)
        else:
            _run_outer(input_c, values, indices, M, N, K, identity)

    if out_provided:
        return _finalize_out(out, values, indices)
    return torch.return_types.cummin((values, indices))


def _resize_for_reuse(t: torch.Tensor, shape) -> None:
    if tuple(t.shape) == tuple(shape):
        return
    t.resize_(shape)


def _finalize_out(out, values, indices):
    uv, ui = out
    if uv.data_ptr() != values.data_ptr():
        uv.copy_(values.view_as(uv))
    if ui.data_ptr() != indices.data_ptr():
        ui.copy_(indices.view_as(ui))
    return torch.return_types.cummin((uv, ui))


def _run_inner(
    x: torch.Tensor, v: torch.Tensor, i: torch.Tensor, M: int, N: int, identity
):
    BLOCK_N = _pick_block_n(N)
    if N <= BLOCK_N:
        cummin_inner_single_kernel[(M,)](
            x,
            v,
            i,
            M,
            N,
            identity,
            BLOCK_N=BLOCK_N,
        )
        return

    num_blocks = triton.cdiv(N, BLOCK_N)
    carry_v = torch.empty((M, num_blocks), dtype=x.dtype, device=x.device)
    carry_i = torch.empty((M, num_blocks), dtype=torch.int64, device=x.device)
    grid = (M, num_blocks)
    cummin_inner_pass1_kernel[grid](
        x,
        v,
        i,
        carry_v,
        carry_i,
        M,
        N,
        num_blocks,
        identity,
        BLOCK_N=BLOCK_N,
    )
    if num_blocks > 2:
        carry_block = max(16, triton.next_power_of_2(num_blocks))
        cummin_inner_carry_scan_kernel[(M,)](
            carry_v,
            carry_i,
            M,
            num_blocks,
            identity,
            BLOCK=carry_block,
        )
    cummin_inner_pass2_kernel[grid](
        v,
        i,
        carry_v,
        carry_i,
        M,
        N,
        num_blocks,
        BLOCK_N=BLOCK_N,
    )


def _run_outer(
    x: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    M_pre: int,
    N: int,
    M_post: int,
    identity,
):
    def grid(meta):
        return (
            M_pre,
            triton.cdiv(M_post, meta["BLOCK_M_POST"]),
        )

    cummin_outer_1d_kernel[grid](
        x,
        v,
        i,
        M_pre,
        N,
        M_post,
        identity,
    )
