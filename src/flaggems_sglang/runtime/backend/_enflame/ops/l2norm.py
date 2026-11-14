# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP32 last-dimension L2 normalization with grouped rows."""
import torch
import triton
import triton.language as tl


def _install_l2norm_gcuir_compat():
    """Scoped GCUIR compat for this kernel family (see
    compiler_issues/sglang_l2norm_masked2d_slice_pad_start.md).

    The process-global dialect shim installed by other enflame ops
    (extend_attention) strips the trailing zero operandSegmentSizes segment
    from every ``memref_ext.slice_*`` line. The current external
    gcu-compiler-opt however requires the full 8-segment (trailing zero)
    form for ``memref_ext.slice_pad_start`` emitted for row-masked 2D tile
    loads, so first-compiles of the row-batched kernel fail with
    "size mismatch in attribute conversion: 7 vs 8". Retry those compiles
    (and only those, matched on this kernel's symbol) straight against the
    original external tool with the untouched IR; fallback-only so every
    already-compilable kernel keeps its exact code path.
    """
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_l2norm_gcuir_compat", False):
        return
    _prev = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if "@_norm" not in text:
                raise first_error
            try:
                return toolkit._run_command("gcu-compiler-opt", text, *args)
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._l2norm_gcuir_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_l2norm_gcuir_compat()  # DISABLED 2026-11-11 重放剥离: 私有 GCUIR shim 按现行策略禁用(5dc28b9);
# slice_pad_start 7vs8 系旧工具链缺陷,P01a 定案后 fresh 编译无需垫片(bmm_chunk 同批先例)


@triton.jit
def _norm(
    X,
    Out,
    ROWS,
    D: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN_COLS: tl.constexpr,
):
    row = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    col = tl.arange(0, BLOCK)
    base = row * XS0
    valid = row[:, None] < ROWS
    if EVEN_COLS:
        x = tl.load(X + base[:, None] + col[None, :] * XS1, valid, other=0).to(
            tl.float32
        )
    else:
        valid = valid & (col[None, :] < D)
        x = tl.load(X + base[:, None] + col[None, :] * XS1, valid, other=0).to(
            tl.float32
        )
    inv = tl.rsqrt(tl.sum(x * x, 1) + EPS)
    tl.store(Out + row[:, None] * D + col[None, :], x * inv[:, None], valid)


def _launch(x, eps=1e-6, group=4, warps=4):
    if x.ndim == 0 or x.shape[-1] <= 0:
        raise ValueError("x must have a positive final dimension")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    width = x.shape[-1]
    rows = x.numel() // width
    if rows:
        flat = x.reshape(rows, width)
        block = triton.next_power_of_2(width)
        _norm[(triton.cdiv(rows, group),)](
            flat,
            out,
            rows,
            width,
            float(eps),
            flat.stride(0),
            flat.stride(1),
            group,
            block,
            block == width,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return out


def _prev_pow2(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 1


def l2norm(x, eps=1e-6):
    width = x.shape[-1]
    rows = x.numel() // width if width else 0
    even = width > 0 and (width & (width - 1)) == 0
    if even and width <= 512:
        # Even-column fast path: no column mask, so large row batches compile
        # (the dual row+column mask only survives GROUP<=8 at num_warps=4).
        if rows >= 2048:
            group, warps = 128, 4
        elif rows >= 64:
            group, warps = min(64, max(16, _prev_pow2(rows // 32))), 2
        elif rows >= 4:
            group, warps = 4, 4
        else:
            group, warps = 1, 4
    else:
        group = (
            8
            if rows >= 64 and width <= 512
            else 4 if rows >= 4 and width <= 512 else 1
        )
        warps = 4 if width <= 4096 else 8
    return _launch(x, eps, group=group, warps=warps)


__all__ = ["l2norm"]
