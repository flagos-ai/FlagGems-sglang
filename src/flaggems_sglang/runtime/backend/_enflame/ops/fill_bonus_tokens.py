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

"""fill_bonus_tokens (speculative bonus-token gather) -- Enflame GCU specialization.

torch_gcu narrows device int64 tensors to packed int32 storage unless
TORCH_GCU_ENABLE_INT64_AND_UINT64 is set, so a kernel that addresses them
with the declared int64 width reads every other physical word and silently
returns zeros. Token ids and accept lengths always fit int32, so the kernels
run entirely in the int32 word domain and one element maps to one word.

With packed storage and an even accept_stride the fast kernel deliberately
uses the interleaving: it loads S/2 int64 lane pairs per row and splits them
into the low/high halves -- GCU issues 8-byte-element loads measurably faster
than 4-byte ones here. True 64-bit storage and odd strides fall back to an
element-domain kernel: int64 lanes narrowed to the low word (true 64-bit
storage) or plain int32 words (packed / native int32); the old stride-2
int32-word addressing fails first-build compilation from 4096 lanes upward.
Both kernels use select-store semantics: exactly one lane per row matches
``col == length - 1`` and that lane alone stores, matching the generic
``_selected`` arm; accept lengths outside [1, S] leave the output word
untouched (the torch reference itself is undefined there). All addressing
stays 1D: any 2D broadcast-offset tile load/store form still fails to
compile on first build (compiler_issues/sglang_fill_bonus_tokens.md family).

Row batching (2026-11-08 perf iteration): one program per row costs a fixed
~2.6 us per 48-program wave (measured wave ladder; sip_count=24, 48 resident
programs), so grid=(n,) pays ~47 ns/row -- n=512 ran 27.6 us, slower than the
torch reference. Each program now covers Y consecutive rows as ONE flat
[Y*lanes_per_row]-lane tile: the src load is a single affine 1D load, per-lane
lengths come from a small gather (lens + row, row = c // lanes_per_row -- each
row's word is re-read lanes_per_row times, all L2 hits), and the select drops
the per-row tl.sum reduction trees entirely. Y = next_pow2(n/16) for n <= 256
else next_pow2(n/32), capped at 512 (empirical sweet spot 16-32 resident
blocks, single wave); num_warps=2 from Y >= 64 (>= 256 lanes) onward.

Hard limits encoded in _launch_geometry: the per-program lane tile is capped
at 2048 because masked tile loads silently corrupt lanes beyond 2048 at 4096
lanes (compiler_issues/sglang_fill_bonus_tokens.md), and grid.x is kept at
65535 by spilling excess blocks into grid.y (block = pid_y * 65535 + pid_x).
GRID2D launches always take the masked path: the spill rounds blocks up, so
dead programs would otherwise issue unmasked far-OOB reads (device abort at
40M rows) even when n % Y == 0. N is not specialized in the fallback kernel
(one binary per geometry, no n-dependent variant churn) and the total lane
count is guarded host-side to stay within int32 addressing.

A tl.reshape(tl.broadcast_to(...)) variant that expands the [Y] length vector
instead of gathering was rejected: +2 us at small n and it fails to compile on
first build at num_warps >= 2 (memref_ext.slice_start 7 vs 6, same 20:22
toolchain regression family -- compiler_issues/sglang_fill_bonus_tokens.md).

num_warps for the words kernel was retuned for true 64-bit storage (the
TORCH_GCU_ENABLE_INT64_AND_UINT64=1 default since 2026-11-11): two warps once
the per-program lane tile reaches 32 lanes (-7.9% at n=64, -26.3% at n=4096,
neutral at n=512); smaller tiles regress (+3.5% at n=8) and stay single-warp.
"""

import torch
import triton
import triton.language as tl

_LANE_TILE_CAP = 2048  # masked tile loads silently corrupt lanes above this


@triton.jit
def _fill_bonus_tokens_paired(
    src_ptr,  # token ids, packed 4B words; read as int64 lane pairs
    lens_ptr,  # accept lengths, packed 4B words
    out_ptr,  # bonus-token output, packed 4B words
    N,  # rows
    Y: tl.constexpr,  # rows covered per program
    Q: tl.constexpr,  # int64 lane pairs per row (S // 2)
    L: tl.constexpr,  # lanes per program (Y * Q)
    L2: tl.constexpr,  # pow2 >= L
    EVEN_ROWS: tl.constexpr,  # n % Y == 0: elide the row-bound mask
    GRID2D: tl.constexpr,  # blocks spilled to grid.y: block = pid_y*65535+pid_x
):
    lens = lens_ptr.to(tl.pointer_type(tl.int32))
    out = out_ptr.to(tl.pointer_type(tl.int32))
    src64 = src_ptr.to(tl.pointer_type(tl.int64))
    if GRID2D:
        block = tl.program_id(1) * 65535 + tl.program_id(0)
    else:
        block = tl.program_id(0)
    c = tl.arange(0, L2)
    keep_lane = c < L
    r = c // Q
    j = c - r * Q
    row = block * Y + r
    colw = j * 2
    base = block * L
    if EVEN_ROWS:
        # keep_lane still guards lanes >= L: with L2 > L their r >= Y and the
        # lens address would run past this program's rows (OOB on the last one).
        # Masked addresses are additionally clamped to element 0: predicated-off
        # lanes must not leave speculative far-OOB addresses on the wire.
        row_c = tl.where(keep_lane, row, 0)
        length = tl.load(lens + row_c, mask=keep_lane, other=1)
        pairs = tl.load(src64 + tl.where(keep_lane, base + c, 0), mask=keep_lane, other=0)
        m = keep_lane
    else:
        keep = keep_lane & (row < N)
        row_c = tl.where(keep, row, 0)
        length = tl.load(lens + row_c, mask=keep, other=1)
        pairs = tl.load(src64 + tl.where(keep, base + c, 0), mask=keep, other=0)
        m = keep
    lo = pairs & 0xFFFFFFFF
    hi = (pairs >> 32) & 0xFFFFFFFF
    target = length - 1
    me = colw == target
    mo = colw + 1 == target
    value = tl.where(me, lo, tl.where(mo, hi, 0))
    tl.store(out + row_c, value.to(tl.int32), mask=(me | mo) & m)


@triton.jit(do_not_specialize=["N"])
def _fill_bonus_tokens_words(
    src_ptr,  # token ids; int64 lanes (true 64-bit storage) or int32 words
    lens_ptr,  # accept lengths; same treatment
    out_ptr,  # bonus-token output; same treatment
    N,  # rows
    Y: tl.constexpr,  # rows covered per program
    S: tl.constexpr,  # accept_stride: logical row width
    L: tl.constexpr,  # element lanes per program (Y * S)
    L2: tl.constexpr,  # pow2 >= L
    SRC_I64: tl.constexpr,  # true 64-bit src storage: int64-lane loads
    LENS_I64: tl.constexpr,  # true 64-bit lens storage
    OUT_I64: tl.constexpr,  # true 64-bit out storage (zero-extended store)
    EVEN_ROWS: tl.constexpr,  # n % Y == 0: elide the row-bound mask
    GRID2D: tl.constexpr,  # blocks spilled to grid.y: block = pid_y*65535+pid_x
):
    # Element-domain addressing: a stride-2 int32-word tile (the old
    # `(offset) * WORD` form) lowers to a strided memref view that fails to
    # compile on first build from L2 >= 4096 lanes (memref_ext.slice_start
    # 7 vs 6, 20:22 regression family -- compiler_issues/
    # sglang_fill_bonus_tokens.md). True 64-bit tensors are therefore
    # addressed as int64 lanes (stride 1) and narrowed to the low word,
    # which is bit-identical to reading the low int32 word.
    c = tl.arange(0, L2)
    keep_lane = c < L
    r = c // S
    j = c - r * S
    if GRID2D:
        block = tl.program_id(1) * 65535 + tl.program_id(0)
    else:
        block = tl.program_id(0)
    row = block * Y + r
    base = block * L
    # Masked lanes (tail rows, dead GRID2D blocks, pad lanes) get their load
    # addresses clamped to element 0: predicated-off accesses must not leave
    # speculative far-OOB addresses on the wire (device abort at 20M rows).
    if EVEN_ROWS:
        # keep_lane still guards lanes >= L: with L2 > L their r >= Y and the
        # lens address would run past this program's rows (OOB on the last one).
        row_c = tl.where(keep_lane, row, 0)
        off_c = tl.where(keep_lane, base + c, 0)
        if LENS_I64:
            lens64 = lens_ptr.to(tl.pointer_type(tl.int64))
            length = tl.load(lens64 + row_c, mask=keep_lane, other=1).to(tl.int32)
        else:
            lens32 = lens_ptr.to(tl.pointer_type(tl.int32))
            length = tl.load(lens32 + row_c, mask=keep_lane, other=1)
        if SRC_I64:
            src64 = src_ptr.to(tl.pointer_type(tl.int64))
            values = tl.load(src64 + off_c, mask=keep_lane, other=0).to(
                tl.int32
            )
        else:
            src32 = src_ptr.to(tl.pointer_type(tl.int32))
            values = tl.load(src32 + off_c, mask=keep_lane, other=0)
        m = keep_lane
    else:
        keep = keep_lane & (row < N)
        row_c = tl.where(keep, row, 0)
        off_c = tl.where(keep, base + c, 0)
        if LENS_I64:
            lens64 = lens_ptr.to(tl.pointer_type(tl.int64))
            length = tl.load(lens64 + row_c, mask=keep, other=1).to(tl.int32)
        else:
            lens32 = lens_ptr.to(tl.pointer_type(tl.int32))
            length = tl.load(lens32 + row_c, mask=keep, other=1)
        if SRC_I64:
            src64 = src_ptr.to(tl.pointer_type(tl.int64))
            values = tl.load(src64 + off_c, mask=keep, other=0).to(tl.int32)
        else:
            src32 = src_ptr.to(tl.pointer_type(tl.int32))
            values = tl.load(src32 + off_c, mask=keep, other=0)
        m = keep
    target = length - 1
    match = (j == target) & m
    value = tl.where(match, values, 0)
    if OUT_I64:
        out64 = out_ptr.to(tl.pointer_type(tl.int64))
        tl.store(out64 + row_c, value.to(tl.int64), mask=match)
    else:
        out32 = out_ptr.to(tl.pointer_type(tl.int32))
        tl.store(out32 + row_c, value, mask=match)


def _probe_i64_word_stride(device):
    """Return the int32 word stride of one int64 element on this backend.

    Probes by writing a known small value through the int64 tensor and
    reading the int32 view: true 64-bit little-endian storage puts the value
    in word 0 and zero in word 1 (stride 2); packed-32-bit storage puts
    consecutive values in consecutive words (stride 1).
    """
    probe = torch.zeros(4, dtype=torch.int64, device=device)
    probe[2] = 7
    words = probe.view(torch.int32).tolist()
    if words[4] == 7 and words[5] == 0:
        return 2
    if words[2] == 7:
        return 1
    return 2


_I64_WORD_STRIDE = None  # int or None until the first int64 call


def _i64_word_stride(device):
    global _I64_WORD_STRIDE
    if _I64_WORD_STRIDE is None:
        _I64_WORD_STRIDE = _probe_i64_word_stride(device)
    return _I64_WORD_STRIDE


def _launch_geometry(n, lanes_per_row):
    """Return (rows per program Y, grid tuple, GRID2D flag).

    Y targets 16-32 resident blocks (one 48-program wave, empirically optimal
    from 1..70000 rows), is capped at 512 rows and at a 2048-lane per-program
    tile (masked tile loads silently corrupt lanes beyond 2048 at 4096 --
    compiler_issues/sglang_fill_bonus_tokens.md). grid.x stays within 65535
    by spilling excess blocks into grid.y.
    """
    div = 16 if n <= 256 else 32
    y = triton.next_power_of_2(max(n // div, 1))
    if y > 512:
        y = 512
    lane_cap = 1
    while lane_cap * 2 * lanes_per_row <= _LANE_TILE_CAP:
        lane_cap *= 2
    if y > lane_cap:
        y = lane_cap
    blocks = triton.cdiv(n, y)
    if blocks <= 65535:
        return y, (blocks,), False
    return y, (65535, triton.cdiv(blocks, 65535)), True


def fill_bonus_tokens(accept_tokens, accept_lens, bonus_tokens, accept_stride):
    n = accept_lens.shape[0]
    out = torch.empty(n, dtype=bonus_tokens.dtype, device=accept_tokens.device)
    if not n:
        return out
    src = accept_tokens.contiguous()
    lens = accept_lens.contiguous()
    s = int(accept_stride)
    if src.dtype not in (torch.int64, torch.int32):
        src = src.to(torch.int32)
    if lens.dtype not in (torch.int64, torch.int32):
        lens = lens.to(torch.int32)
    if n * max(s, 1) >= 2**31:
        raise ValueError(
            "fill_bonus_tokens: row count times stride exceeds int32 lanes"
        )
    packed = _i64_word_stride(accept_tokens.device)
    staging = out.dtype not in (torch.int64, torch.int32)
    if staging:
        target = torch.empty(n, dtype=torch.int32, device=out.device)
    else:
        target = out
    if packed == 1 and s % 2 == 0:
        q = s // 2
        y, grid, grid2d = _launch_geometry(n, q)
        lanes = y * q
        _fill_bonus_tokens_paired[grid](
            src,
            lens,
            target,
            n,
            Y=y,
            Q=q,
            L=lanes,
            L2=triton.next_power_of_2(max(lanes, 1)),
            # GRID2D rounds blocks up to gridx*gridy: dead programs exist and
            # must take the masked (address-clamped) path even when n % y == 0.
            EVEN_ROWS=(n % y == 0 and not grid2d),
            GRID2D=grid2d,
            num_warps=2 if y >= 64 else 1,
        )
    else:
        y, grid, grid2d = _launch_geometry(n, s)
        lanes = y * s
        _fill_bonus_tokens_words[grid](
            src,
            lens,
            target,
            n,
            Y=y,
            S=s,
            L=lanes,
            L2=triton.next_power_of_2(max(lanes, 1)),
            SRC_I64=(src.dtype == torch.int64 and packed == 2),
            LENS_I64=(lens.dtype == torch.int64 and packed == 2),
            OUT_I64=(not staging and out.dtype == torch.int64 and packed == 2),
            EVEN_ROWS=(n % y == 0 and not grid2d),
            GRID2D=grid2d,
            # Retuned for true 64-bit storage (env-on default since 2026-11-11):
            # two warps hide load latency once the per-program lane tile reaches
            # 32 lanes (n=64 -7.9%, n=4096 -26.3%); smaller tiles regress (+3.5%
            # at n=8), so they stay single-warp. Mask-elision (NO_PAD arm for
            # pow2 L / EVEN_ROWS) measured within +-0.4% of the masked kernel
            # and was rejected for complexity.
            num_warps=2 if lanes >= 32 else 1,
        )
    if staging:
        out.copy_(target)
    return out


__all__ = ["fill_bonus_tokens"]
