import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    X,
    O,
    n_cols,
    n_tiles,
    in_row_stride,
    out_row_stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // n_tiles
    tile = pid % n_tiles
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * in_row_stride
    x1 = tl.load(X + base + offs, mask=mask, other=0.0)
    x3 = tl.load(X + base + n_cols + offs, mask=mask, other=0.0)
    x1f = x1.to(tl.float32)
    y = (x1f * (1.0 / (1.0 + tl.exp(-x1f)))) * x3.to(tl.float32)
    tl.store(
        O + row * out_row_stride + offs, y.to(O.dtype.element_ty), mask=mask
    )


def run(hidden_states):
    x = hidden_states
    d = x.shape[-1] // 2
    out_shape = tuple(x.shape[:-1]) + (d,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if d == 0:
        return out

    if not x.is_contiguous():
        x = x.contiguous()

    if x.dim() <= 1:
        n_rows = 1
        in_row_stride = 2 * d
    else:
        n_rows = x.numel() // (2 * d)
        in_row_stride = 2 * d

    BLOCK = 2048
    n_tiles = triton.cdiv(d, BLOCK)
    grid = (n_rows * n_tiles,)
    _silu_and_mul_kernel[grid](
        x,
        out,
        d,
        n_tiles,
        in_row_stride,
        d,
        BLOCK=BLOCK,
    )
    return out


silu_and_mul = run
__all__ = ["silu_and_mul"]
