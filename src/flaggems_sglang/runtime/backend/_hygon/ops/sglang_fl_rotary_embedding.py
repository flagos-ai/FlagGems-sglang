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

"""Fused in-place RoPE (SGLang apply_rope_with_cos_sin_cache_inplace semantics).

One Triton launch per run() call for the common case (query and key share dtype
and row count). The grid is (token blocks, NBQ + NBK); programs in the first NBQ
head-block slots rotate query, the rest rotate key. Removing the second launch
removed one per-invocation dispatcher interval, which the round-1 profile showed
dominates the launch-bound timing shapes.

The cos/sin gather is a tensor-of-pointer load along the head axis (stride 0),
so the [BLOCK_T, HALF] broadcast never becomes a shared-memory layout
conversion: the round-1 profile showed the 2-D-load-then-broadcast form emitting
an LDS round trip (ds_write/ds_read + s_barrier), while a target-side ISA census
showed the stride-0 form compiling to zero LDS traffic with 128-bit cache loads.

The activation accesses carry streaming hints (.cg load, .cs store): this is a
pure streaming write-in-place pass whose results are not re-read inside the
kernel, and a same-job target-side A/B measured b4-s4096 at 147.7 us with the
hints versus 154.2 us without (-4.2%), while the small shapes were unaffected.

Wrapper requirements honoured here:
  * returns None, writes the rotated values back into the caller's query/key
    buffers (no output allocation, no .copy_(), no intermediate tensor),
  * rotary_dim comes from cos_sin_cache.shape[-1] (cos in the first half of the
    last dim, sin in the second half),
  * rotation is computed in float32 and stored back in the tensors' own dtype,
  * lanes past rotary_dim inside a head are left completely untouched (they are
    never loaded nor stored),
  * is_neox selects between the two structurally different lane pairings and is
    a constexpr so each gets its own vectorised path.
"""

import triton
import triton.language as tl


@triton.jit
def _rope_rotate(
    x_ptr,
    base,
    m,
    c,
    s,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NEOX: tl.constexpr,
):
    """Rotate the leading D lanes of BLOCK_H heads of BLOCK_T token rows."""
    if NEOX:
        # neox pairs lane p with lane p + HALF inside the head.
        o1 = base + tl.arange(0, HALF)[None, None, :]
        o2 = o1 + HALF
        x1 = tl.load(x_ptr + o1, mask=m, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )
        x2 = tl.load(x_ptr + o2, mask=m, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )
        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s
        tl.store(
            x_ptr + o1,
            y1.to(x_ptr.dtype.element_ty),
            mask=m,
            cache_modifier=".cs",
        )
        tl.store(
            x_ptr + o2,
            y2.to(x_ptr.dtype.element_ty),
            mask=m,
            cache_modifier=".cs",
        )
    else:
        # interleaved: lane 2p pairs with lane 2p + 1, so the whole head is one
        # contiguous run that is split/rejoined in registers.
        o = base + tl.arange(0, D)[None, None, :]
        x = tl.load(x_ptr + o, mask=m, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )
        x1, x2 = tl.split(tl.reshape(x, (BLOCK_T, BLOCK_H, HALF, 2)))
        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s
        y = tl.reshape(tl.join(y1, y2), (BLOCK_T, BLOCK_H, D))
        tl.store(
            x_ptr + o,
            y.to(x_ptr.dtype.element_ty),
            mask=m,
            cache_modifier=".cs",
        )


@triton.jit
def _rope_fused_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cs_ptr,
    num_tokens,
    pos_len,
    NBQ,
    NHQ,
    NHK,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NEOX: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid = tl.program_id(1)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = (t < num_tokens) & (t < pos_len)
    pos = tl.load(pos_ptr + t, mask=tmask, other=0)

    if pid < NBQ:
        blk = pid
        NH = NHQ
        RS = NHQ * H
        x_ptr = q_ptr
    else:
        blk = pid - NBQ
        NH = NHK
        RS = NHK * H
        x_ptr = k_ptr

    hp = blk * BLOCK_H + tl.arange(0, BLOCK_H)
    m = tmask[:, None, None] & (hp < NH)[None, :, None]
    base = t[:, None, None] * RS + hp[None, :, None] * H

    p = tl.arange(0, HALF)[None, None, :]
    # cos in columns [0, HALF), sin in columns [HALF, D) of the cache row; the
    # head axis contributes a zero stride so no layout conversion is needed.
    row = pos[:, None, None].to(tl.int32) * D
    cm = tmask[:, None, None]
    c = tl.load(cs_ptr + row + p, mask=cm, other=0.0)
    s = tl.load(cs_ptr + row + p + HALF, mask=cm, other=0.0)

    _rope_rotate(x_ptr, base, m, c, s, H, D, HALF, BLOCK_T, BLOCK_H, NEOX)


@triton.jit
def _rope_single_kernel(
    x_ptr,
    pos_ptr,
    cs_ptr,
    num_tokens,
    pos_len,
    NH,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NEOX: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = (t < num_tokens) & (t < pos_len)
    pos = tl.load(pos_ptr + t, mask=tmask, other=0)

    hp = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    m = tmask[:, None, None] & (hp < NH)[None, :, None]
    base = t[:, None, None] * (NH * H) + hp[None, :, None] * H

    p = tl.arange(0, HALF)[None, None, :]
    row = pos[:, None, None].to(tl.int32) * D
    cm = tmask[:, None, None]
    c = tl.load(cs_ptr + row + p, mask=cm, other=0.0)
    s = tl.load(cs_ptr + row + p + HALF, mask=cm, other=0.0)

    _rope_rotate(x_ptr, base, m, c, s, H, D, HALF, BLOCK_T, BLOCK_H, NEOX)


def _plan(n, nh_max, half):
    """Tile plan chosen from the target-side sweeps: four token rows per program,
    a head block carrying about 1024 lanes, and the matching warp count."""
    bt = 4
    bh = max(1, 1024 // (half * bt))
    if bh > nh_max:
        bh = nh_max
    bh = 1 << max(0, bh.bit_length() - 1)
    lanes = bt * bh * half
    if lanes >= 256:
        nw = 4
    elif lanes >= 64:
        nw = 2
    else:
        nw = 1
    return bt, bh, nw


def _launch_single(x, pos, pos_len, cs, H, D, HALF, neox):
    n = int(x.shape[0])
    nh = int(x.shape[1]) // H
    if n <= 0 or nh <= 0:
        return
    bt, bh, nw = _plan(n, nh, HALF)
    _rope_single_kernel[(triton.cdiv(n, bt), triton.cdiv(nh, bh))](
        x,
        pos,
        cs,
        n,
        pos_len,
        nh,
        H=H,
        D=D,
        HALF=HALF,
        BLOCK_T=bt,
        BLOCK_H=bh,
        NEOX=neox,
        num_warps=nw,
    )


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    H = int(head_size)
    D = int(cos_sin_cache.shape[-1])
    HALF = D // 2
    if H <= 0 or D <= 0 or D % 2 != 0 or HALF > H:
        # Nothing safely rotatable through the fused path.
        return query, key
    neox = bool(is_neox)
    pos = positions.reshape(-1)
    pos_len = int(pos.numel())

    n = int(query.shape[0])
    nhq = int(query.shape[1]) // H
    nhk = int(key.shape[1]) // H
    if n <= 0 or nhq <= 0 or nhk <= 0:
        return query, key

    if query.dtype == key.dtype and int(key.shape[0]) == n:
        # One launch for both tensors (the common, and always-experimented, case).
        nh_max = nhq if nhq < nhk else nhk
        bt, bh, nw = _plan(n, nh_max, HALF)
        nbq = triton.cdiv(nhq, bh)
        nbk = triton.cdiv(nhk, bh)
        _rope_fused_kernel[(triton.cdiv(n, bt), nbq + nbk)](
            query,
            key,
            pos,
            cos_sin_cache,
            n,
            pos_len,
            nbq,
            nhq,
            nhk,
            H=H,
            D=D,
            HALF=HALF,
            BLOCK_T=bt,
            BLOCK_H=bh,
            NEOX=neox,
            num_warps=nw,
        )
    else:
        # Mixed dtypes / row counts: one launch per tensor.
        _launch_single(query, pos, pos_len, cos_sin_cache, H, D, HALF, neox)
        _launch_single(key, pos, pos_len, cos_sin_cache, H, D, HALF, neox)
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
