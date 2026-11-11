# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-token-group FP8 quantization.

One program handles several consecutive groups; each group reduces its own
absolute maximum, derives ``scale = amax / 448`` from it and multiplies by the
reciprocal ``448 / amax`` before clamping into the finite E4M3 range.

The two shapes of that reciprocal matter: sglang's kernel computes
``scale_inv = amax * (1 / 448)`` and multiplies by ``448 / amax`` -- it never
divides each element by the scale -- and it floors the absolute maximum at
``1e-10`` so an all-zero group still yields a finite scale instead of a
division by zero. Both are reproduced here, and the emitted scale is the small
value ``amax / 448``, which is what the caller receives from sglang.

Structurally this mirrors ``per_token_group_quant_int8``, the sibling already
in this repository: same group-per-program packing, same ``MAX_ELEMS`` cap, and
a masked variant for a group size that is not a power of two. Only the
quantization range and the destination dtype differ. The literals stay inline
in the kernels rather than in module constants because a ``@triton.jit`` body
cannot read a plain module global, and the sibling does the same.
"""

import torch
import triton
import triton.language as tl

# Upper bound on the elements a single program holds in registers, matching the
# int8 sibling so both operators tile the same way.
MAX_ELEMS = 2048


@triton.jit
def _quant_exact(
    x_ptr, q_ptr, s_ptr, GROUP_SIZE: tl.constexpr, GPP: tl.constexpr
):
    g = tl.program_id(0) * GPP + tl.arange(0, GPP)
    offs = g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x = tl.load(x_ptr + offs).to(tl.float32)
    # 448 is the finite maximum of the saturating E4M3 variant sglang emits;
    # 1e-10 floors an all-zero group so its scale stays finite.
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    inv = 448.0 / amax
    q = tl.clamp(x * inv[:, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(q_ptr + offs, q)
    tl.store(s_ptr + g, amax / 448.0)


@triton.jit
def _quant_masked(
    x_ptr,
    q_ptr,
    s_ptr,
    NUM_GROUPS,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GPP: tl.constexpr,
):
    g = tl.program_id(0) * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, BLOCK)
    gmask = g < NUM_GROUPS
    mask = gmask[:, None] & (lanes[None, :] < GROUP_SIZE)
    offs = g[:, None] * GROUP_SIZE + lanes[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    inv = 448.0 / amax
    q = tl.clamp(x * inv[:, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(s_ptr + g, amax / 448.0, mask=gmask)


def per_token_group_quant_fp8(x, group_size, dtype=torch.float8_e4m3fn):
    """Quantize groups along the last dimension of ``x`` to FP8.

    Args:
        x: input tensor of any floating dtype Triton can read. The last
            dimension must be a multiple of ``group_size``.
        group_size: number of consecutive elements sharing one scale.
        dtype: FP8 destination dtype; only float8_e4m3fn is supported.

    Returns:
        ``(quantized, scales)``: ``quantized`` has ``x``'s shape and
        ``scales`` has ``x.shape[:-1] + (x.shape[-1] // group_size,)``.
    """
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(
            f"group_size must be a positive int, got {group_size!r}"
        )
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} is not a multiple of "
            f"group_size {group_size}"
        )
    if dtype != torch.float8_e4m3fn:
        raise ValueError(f"only float8_e4m3fn is supported, got {dtype}")

    shape = x.shape
    cols = shape[-1]
    num_groups = x.numel() // group_size
    if num_groups == 0:
        # Nothing to quantize; a zero grid is rejected by Triton.
        q = torch.empty(shape, device=x.device, dtype=dtype)
        s = torch.empty(
            shape[:-1] + (cols // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
        return q, s

    block = triton.next_power_of_2(group_size)
    gpp = 1
    while gpp * 2 * block <= MAX_ELEMS and num_groups % (gpp * 2) == 0:
        gpp *= 2
    warps = min(max(block * gpp // 256, 1), 8)

    q = torch.empty(shape, device=x.device, dtype=dtype)
    s = torch.empty(
        shape[:-1] + (cols // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(num_groups, gpp), 1, 1)
    if block == group_size:
        _quant_exact[grid](
            x,
            q,
            s,
            GROUP_SIZE=group_size,
            GPP=gpp,
            num_warps=warps,
            num_stages=1,
        )
    else:
        _quant_masked[grid](
            x,
            q,
            s,
            num_groups,
            GROUP_SIZE=group_size,
            BLOCK=block,
            GPP=gpp,
            num_warps=warps,
            num_stages=1,
        )
    return q, s


__all__ = ["per_token_group_quant_fp8"]
