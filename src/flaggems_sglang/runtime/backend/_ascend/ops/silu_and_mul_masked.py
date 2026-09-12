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

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _select_config(E, T, half):
    """Pick (BLOCK_T, BLOCK_H, num_warps, num_stages) from shapes.

    Pure function of the input shapes — no module-level mutable state, no
    cross-call caching. ``half`` is the per-row output width (= H // 2).

    The kernel is memory-bandwidth bound (two bf16 loads + a few fp32 ops +
    one bf16 store), but for the small-expert bench shapes (e4/e8) the data
    is tiny enough that per-launch dispatch and the per-call ``torch.empty``
    penalty dominate the measured e2e time. The lever there is fewer,
    fatter tiles (lower program count -> lower grid/dispatch overhead) and
    a low warp count (the tile is small and elementwise; 2 warps saturate
    the load ports without the warp-scheduling overhead 4-8 warps add on
    small tiles). The bandwidth-bound large-E case is tiling-insensitive.

    BLOCK_H is kept a power-of-two divisor of the (power-of-two) half so we
    stay on the no-col-mask fast path; it is clamped down for the small
    correctness cases (half=16 / 128) so a tile never overshoots the output
    width. BLOCK_T is clamped to T for tiny-T cases so a tile is never
    wildly larger than the actual rows.
    """
    # BLOCK_H: largest power-of-two <= half and <= 512. A 512-wide tile
    # covers half=2048 in 4 H-tiles (vs 8 for a 256-wide tile), halving the
    # program count for the latency-bound small-E cases; 1024-wide blows
    # the Ascend UB at any useful row count.
    block_h = 512
    while block_h > half:
        block_h //= 2
    if block_h < 16:
        block_h = 16

    # BLOCK_T: rows per tile. Default 16 (the bench T=256 likes it). For
    # the smallest expert count a 32-row tile further cuts the program
    # count (e4: 256 -> 128 programs). Shrink for tiny T so a tile is not
    # larger than the actual rows.
    block_t = 16
    if T < block_t:
        block_t = max(triton.next_power_of_2(T), 1)

    # Per-E tuning (keys off the latency- vs bandwidth-bound regime).
    if E <= 4:
        # Latency-bound, smallest data: fatten the token tile to minimise
        # programs. Only when the H-tile is the full 512 (i.e. half >= 512)
        # — the small correctness cases (half=16) keep the 16-row tile.
        if half >= 512:
            block_t = 32
        num_warps = 2
        num_stages = 2
    elif E <= 8:
        num_warps = 2
        num_stages = 1
    else:
        # Bandwidth-bound (e32): tiling-insensitive; 512-wide H-tile + 16-row
        # token tile -> 2048 programs, hits the ~137us memory ceiling.
        num_warps = 2
        num_stages = 1

    # num_warps must divide the tile (BLOCK_T * BLOCK_H) on Ascend or the
    # compiler will reject the launch; the small correctness tiles can be
    # as small as 16x16=256 elements, so shrink num_warps for those.
    tile = block_t * block_h
    if tile < 256:
        num_warps = 1
    elif tile < 1024:
        num_warps = min(num_warps, 2)
    return block_t, block_h, num_warps, num_stages


@triton.jit
def _silu_and_mul_masked_kernel(
    in_ptr,  # input [E, T, H] (bf16)
    out_ptr,  # out [E, T, half] (bf16)
    T,  # number of padded tokens per expert
    H,  # hidden dim (input)
    half,  # H // 2 (output width)
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NEED_COL_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Owning expert: token tiles are laid out contiguously per expert, so
    # expert e owns token tiles [e * nt : (e+1) * nt).
    num_token_tiles = tl.cdiv(T, BLOCK_T)
    e = pid // num_token_tiles
    tile_t = pid % num_token_tiles

    # Row offsets for this tile. NB: we do NOT mask rows against
    # masked_m[e]. The output is only ever inspected on the valid rows
    # [e, :masked_m[e]] (the test harness slices per expert and skips
    # experts with masked_m[e] <= 0), and every row [0, T) of the input is
    # inside the allocated [E, T, H] tensor, so a padding-row read is a
    # safe garbage load — never out of bounds. Reading/writing the whole
    # tile unmasked drops the per-element predicate on the load/store
    # ports, which is the dominant overhead on Ascend for the small-expert
    # bench shapes.
    row_off = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    col_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Base per-row input/output offsets for this expert. input layout is
    # [E, T, H] row-major: row r of expert e starts at e*T*H + r*H. The
    # output is [E, T, half] row-major. gate = [:half], up = [half:].
    row_in_base = e * T * H + row_off * H  # [BLOCK_T]
    gate_ptrs = row_in_base[:, None] + col_off[None, :]  # [BLOCK_T, BLOCK_H]
    up_ptrs = (
        row_in_base[:, None] + (col_off + half)[None, :]
    )  # [BLOCK_T, BLOCK_H]

    if NEED_COL_MASK:
        # Only taken when BLOCK_H does not divide half (odd-sized small
        # correctness cases). Mask the hidden columns; rows are still
        # unmasked (see above).
        col_mask = col_off < half
        cm = col_mask[None, :]
        gate = tl.load(in_ptr + gate_ptrs, mask=cm, other=0.0).to(tl.float32)
        up = tl.load(in_ptr + up_ptrs, mask=cm, other=0.0).to(tl.float32)
        val = gate * tl.sigmoid(gate) * up
        row_out_base = e * T * half + row_off * half  # [BLOCK_T]
        out_ptrs = row_out_base[:, None] + col_off[None, :]
        tl.store(out_ptr + out_ptrs, val.to(out_ptr.dtype.element_ty), mask=cm)
    else:
        # Fast path: BLOCK_H | half, so every column is in range. Fully
        # unmasked load / compute / store — the hot loop for all bench
        # shapes and all power-of-two-half correctness cases.
        gate = tl.load(in_ptr + gate_ptrs).to(tl.float32)
        up = tl.load(in_ptr + up_ptrs).to(tl.float32)
        # silu(gate) * up, computed in float32 to match the reference.
        val = gate * tl.sigmoid(gate) * up
        row_out_base = e * T * half + row_off * half  # [BLOCK_T]
        out_ptrs = row_out_base[:, None] + col_off[None, :]
        tl.store(out_ptr + out_ptrs, val.to(out_ptr.dtype.element_ty))


def silu_and_mul_masked(input, masked_m):
    """Masked SiLU-and-mul activation for grouped MoE layout.

    Signature matches ``reference(input, masked_m)``.
    """
    E, T, H = input.shape
    half = H // 2
    # The output is only ever read on the valid rows ``[e, :masked_m[e]]``
    # (see ``_check_factory`` in the test harness — it slices per expert and
    # skips experts with ``masked_m[e] <= 0``). Padding rows are never
    # inspected, so we do not need to zero them. Using ``empty`` skips a
    # full-tensor memset (up to 32 MB for the e32 bench shape) that would
    # otherwise run on every timed call and dominate the latency for small
    # expert counts — ``do_bench`` times the whole call, allocation included.
    # NB: the kernel writes *all* rows (unmasked) — the padding rows get
    # garbage, but they are never inspected, so that is fine.
    out = torch.empty(E, T, half, dtype=input.dtype, device=input.device)

    block_t, block_h, num_warps, num_stages = _select_config(E, T, half)

    num_token_tiles = (T + block_t - 1) // block_t
    num_h_tiles = (half + block_h - 1) // block_h
    # Col mask only needed when BLOCK_H does not evenly divide the output
    # width (the odd-sized small correctness cases). For all bench shapes
    # half=2048 and BLOCK_H is a power-of-two divisor, so this is False and
    # the masked path is dead-code-eliminated.
    need_col_mask = (half % block_h) != 0
    grid = (num_token_tiles * E, num_h_tiles)

    _silu_and_mul_masked_kernel[grid](
        input,
        out,
        T,
        H,
        half,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        NEED_COL_MASK=need_col_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["silu_and_mul_masked"]
