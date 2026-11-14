# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import torch
import triton
import triton.language as tl

# GCU dispatch calibration (L600/GCU400, do_bench sweep over the 19 benchmark
# shapes, 2026-11-08). The op is a streaming group reduction: the wall is
# per-program fixed cost (reduction tree + dispatch), so the win comes from
# row-batching groups into fat [GPP, BLOCK] tiles instead of many tiny
# programs:
#   * GPP grows with num_groups (sweet spot grid = 16-32 programs), tile
#     capped at 32768 elements, extended to 65536 once grid would pass 128
#     (measured: ng=262144 gpp512 137.8us vs gpp256 151.0us).
#   * tiny tensors (<=32 groups) stay on a 1D single-group kernel at the
#     launch floor; [2, BLOCK] tiles are a measured pathology (7.3us vs
#     4.0us at ng=32), hence the GPP floor of 4 on the row-batch arm.
#   * num_warps: 2 for row-batch tiles, 4 when the machine is CTA-starved
#     (grid <= 16 with >= 16384-lane tiles), 8 for the single arm.
#   * non-divisible group counts keep the fill-rule GPP on the masked arm
#     (big-tile predicate tax ~ 0: ng=332 gpp16-masked 4.8us vs gpp4-exact
#     9.6us); the exact arm only fires when group_size is a power of two
#     and num_groups % GPP == 0 (0.4-6% over masked).
#   * grids stay 1-D and far below the 65535 hardware cap: 2-D grid
#     spreading across the cap is not a rescue (hangs / state corruption,
#     see compiler_issues/grid_hard_limits.md).
_TILE_ELEMS = 32768
_TILE_ELEMS_EXT = 65536
_GRID_EXTEND = 128
_GRID_X_CAP = 65535
_SINGLE_MAX_GROUPS = 32
_MIN_ROW_BATCH = 4


def _prev_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@triton.jit
def _quant_single(
    x_ptr,
    q_ptr,
    s_ptr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    offs = g.to(tl.int64) * GROUP_SIZE + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < GROUP_SIZE
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1.0e-10)
    inv = 127.0 / amax
    q = tl.clamp(x * inv, -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(s_ptr + g, amax / 127.0)


@triton.jit
def _quant_exact(
    x_ptr, q_ptr, s_ptr, GROUP_SIZE: tl.constexpr, GPP: tl.constexpr
):
    pid = tl.program_id(0)
    g = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, GROUP_SIZE)
    offs = g[:, None].to(tl.int64) * GROUP_SIZE + lanes[None, :]
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    inv = 127.0 / amax
    q = tl.clamp(x * inv[:, None], -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q)
    tl.store(s_ptr + g, amax / 127.0)


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
    pid = tl.program_id(0)
    g = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, BLOCK)
    gmask = g < NUM_GROUPS
    mask = gmask[:, None] & (lanes[None, :] < GROUP_SIZE)
    offs = g[:, None].to(tl.int64) * GROUP_SIZE + lanes[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-10)
    inv = 127.0 / amax
    q = tl.clamp(x * inv[:, None], -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(s_ptr + g, amax / 127.0, mask=gmask)


def per_token_group_quant_int8(x, group_size, dtype=torch.int8):
    if not x.is_contiguous():
        x = x.contiguous()
    shape = x.shape
    cols = shape[-1]
    num_groups = x.numel() // group_size
    block = triton.next_power_of_2(group_size)
    q = torch.empty(shape, device=x.device, dtype=dtype)
    s = torch.empty(
        shape[:-1] + (cols // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    if num_groups == 0:
        return q, s

    if num_groups <= _SINGLE_MAX_GROUPS:
        _quant_single[(num_groups,)](
            x,
            q,
            s,
            GROUP_SIZE=group_size,
            BLOCK=block,
            num_warps=8 if block <= 1024 else 4,
            num_stages=1,
        )
        return q, s

    gpp = _prev_pow2(max(1, num_groups // 16))
    gpp = max(_MIN_ROW_BATCH, gpp)
    cap = max(1, _TILE_ELEMS // block)
    cap_ext = max(cap, _TILE_ELEMS_EXT // block)
    gpp = min(gpp, cap)
    if triton.cdiv(num_groups, gpp) > _GRID_EXTEND and cap_ext > gpp:
        gpp = min(cap_ext, gpp * 2)
    if triton.cdiv(num_groups, gpp) > _GRID_X_CAP:
        gpp = max(
            gpp, triton.next_power_of_2(triton.cdiv(num_groups, _GRID_X_CAP))
        )

    grid = (triton.cdiv(num_groups, gpp),)
    lanes = gpp * block
    if grid[0] <= 16 and lanes >= 16384:
        warps = 4
    else:
        warps = 2
    if block == group_size and num_groups % gpp == 0:
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


__all__ = ["per_token_group_quant_int8"]
