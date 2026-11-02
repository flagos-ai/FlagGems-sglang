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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _state_scatter_tile_kernel(
    dst_ptr,  # [L, C, *S]      in-place destination
    src_ptr,  # [L, R, D, *S]   gather source
    dst_idx_ptr,  # [R] int32
    step_ptr,  # [R] int32
    R,  # requests
    C,  # cache size
    D,  # draft (steps per request)
    S,  # numel of trailing state dims
    L,  # layers
    NR: tl.constexpr,  # requests per program (power of 2)
    BLOCK: tl.constexpr,
    EVEN_R: tl.constexpr,  # R % NR == 0: compile out the request mask
    EVEN: tl.constexpr,  # S % BLOCK == 0: compile out the bounds mask
    WIDE: tl.constexpr,  # offsets need 64-bit index arithmetic
):
    grp = tl.program_id(0)
    layer = tl.program_id(1)
    blk = tl.program_id(2)

    reqs = grp * NR + tl.arange(0, NR)
    if EVEN_R:
        reqm = tl.full((NR,), True, tl.int1)
    else:
        reqm = reqs < R
    live = reqm & ((layer + 1) <= L)
    step_v = tl.load(step_ptr + reqs, mask=live, other=0)
    dst_v = tl.load(dst_idx_ptr + reqs, mask=live, other=0)

    step_c = tl.maximum(step_v, 0)
    if WIDE:
        src_base = (
            (layer.to(tl.int64) * R + reqs.to(tl.int64)) * D
            + step_c.to(tl.int64)
        ) * S
        dst_base = (layer.to(tl.int64) * C + dst_v.to(tl.int64)) * S
    else:
        src_base = ((layer * R + reqs) * D + step_c) * S
        dst_base = (layer * C + dst_v) * S

    cols = blk * BLOCK + tl.arange(0, BLOCK)
    if EVEN_R:
        row_ok = None
    else:
        row_ok = reqm[:, None]
    if not EVEN:
        inner = cols[None, :] < S
        row_ok = inner if row_ok is None else (row_ok & inner)

    if row_ok is None:
        tile = tl.load(src_ptr + src_base[:, None] + cols[None, :])
    else:
        tile = tl.load(
            src_ptr + src_base[:, None] + cols[None, :], mask=row_ok, other=0
        )
    # Dead requests (step < 0) and padding rows never touch dst.
    store_m = step_v >= 0
    if EVEN_R and EVEN:
        store_m = store_m[:, None]
    else:
        if row_ok is not None:
            store_m = row_ok & store_m[:, None]
        else:
            store_m = store_m[:, None]
        if not EVEN:
            store_m = store_m & (cols[None, :] < S)
    tl.store(dst_ptr + dst_base[:, None] + cols[None, :], tile, mask=store_m)


def state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    """Scatter each live request state into its destination slot in-place."""
    layers, cache, *state_shape = dst.shape
    _, requests, draft, *src_state_shape = src.shape
    if requests == 0:
        return dst
    state_numel = 1
    for d in state_shape:
        state_numel *= d

    # One full-row block with one warp copies best (deep per-thread
    # pipelining); smaller rows fall back to narrower blocks.
    if state_numel >= 2048:
        BLOCK, num_warps = 2048, 1
    elif state_numel >= 1024:
        BLOCK, num_warps = 1024, 2
    else:
        BLOCK = max(triton.next_power_of_2(max(state_numel, 1)), 32)
        num_warps = 2 if BLOCK >= 256 else 1

    # 32-bit offset arithmetic is safe iff neither tensor spans >= 2**31
    # elements; otherwise fall back to the 64-bit path.
    wide = dst.numel() >= 2**31 or src.numel() >= 2**31

    even = state_numel % BLOCK == 0

    if requests >= 96:
        nr = 4
    elif requests >= 32:
        nr = 2
    else:
        nr = 1
    while nr > 1 and requests % nr != 0:
        nr //= 2
    even_r = requests % nr == 0

    grid = (triton.cdiv(requests, nr), layers, triton.cdiv(state_numel, BLOCK))
    _state_scatter_tile_kernel[grid](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        requests,
        cache,
        draft,
        state_numel,
        layers,
        NR=nr,
        BLOCK=BLOCK,
        EVEN_R=even_r,
        EVEN=even,
        WIDE=wide,
        num_warps=num_warps,
    )
    return dst


__all__ = ["state_scatter_with_mask"]
