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

# GCU launch limits: grid.x <= 65535 (grid.y/z would be <= 255). The generic
# kernels size their 1D grid from the workload (large inputs can request far
# more programs), so the enflame copy caps the launch at this limit and folds
# the surplus programs into a grid-stride loop (STEPS iterations of PROGRAMS
# stride), matching the concat_mla_k precedent.
_GRID_X_MAX = 65535

# L600 has 48 SPs; per-program fixed cost (~50-130ns measured on this op)
# makes many small programs program-count-bound, so the row-batch fast path
# always folds its jobs onto at most this many programs (deep grid-stride
# loop), matching the fixed-grid doctrine.
_FIXED_GRID = 48


@triton.jit
def _rowbatch(
    Q,
    K,
    V,
    O,
    R: tl.constexpr,
    W: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    U: tl.constexpr,
    RB: tl.constexpr,
    JOBS: tl.constexpr,
    STEPS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    # Fast path for standard contiguous q/k/v: one job covers U rows of one
    # (rank, head) pair. All vector addresses are affine in the iota pair
    # (row, col) -- no per-element div/mod, no gathers. The three wide loads
    # are issued before the three stores so independent segments pipeline.
    HD: tl.constexpr = W * L * D
    start = tl.program_id(0)
    d = tl.arange(0, D)[None, :]
    for step in range(STEPS):
        job = start + step * PROGRAMS
        if job < JOBS:
            pair = job // RB
            rb = job % RB
            rank = pair // L
            head = pair % L
            r = rb * U + tl.arange(0, U)[:, None]
            m = r < R
            gh = rank * L + head
            src = r * HD + gh * D + d
            q = tl.load(Q + src, m)
            k = tl.load(K + src, m)
            v = tl.load(V + src, m)
            dst = ((rank * R + r) * L + head) * (3 * D) + d
            tl.store(O + dst, q, m)
            tl.store(O + dst + D, k, m)
            tl.store(O + dst + 2 * D, v, m)


@triton.jit
def _split(
    Q,
    K,
    V,
    O,
    META: tl.constexpr,
    GROUP: tl.constexpr,
    INTERLEAVE: tl.constexpr,
):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    B: tl.constexpr = META[10]
    W: tl.constexpr = H // L
    T: tl.constexpr = triton.cdiv(GROUP * L * D, B)
    pid = tl.program_id(0)
    count: tl.constexpr = W * triton.cdiv(R, GROUP) * T
    if INTERLEAVE:
        kind = pid % 3
        pid = pid // 3
    else:
        kind = pid // count
        pid = pid % count
    p = tl.where(kind == 0, Q, tl.where(kind == 1, K, V))
    s0 = tl.where(kind == 0, META[4], tl.where(kind == 1, META[6], META[8]))
    s1 = tl.where(kind == 0, META[5], tl.where(kind == 1, META[7], META[9]))
    rank = pid // T % W
    x = pid % T * B + tl.arange(0, B)
    row = pid // (W * T) * GROUP + x // (L * D)
    head = x // D % L
    col = x % D
    mask = (row < R) & (x < GROUP * L * D)
    value = tl.load(p + row * s0 + (rank * L + head) * s1 + col, mask)
    dst = ((rank * R + row) * L + head) * (3 * D) + kind * D + col
    tl.store(O + dst, value, mask)


@triton.jit
def _grouped(
    Q,
    K,
    V,
    O,
    META: tl.constexpr,
    GROUP: tl.constexpr,
    HINT: tl.constexpr,
    STREAM: tl.constexpr,
    STEPS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    B: tl.constexpr = META[10]
    if R * H * D > 0:
        W: tl.constexpr = H // L
        TILES: tl.constexpr = (GROUP * L * D + B - 1) // B
        JOBS: tl.constexpr = W * TILES * ((R + GROUP - 1) // GROUP)
        start = tl.program_id(0)
        for step in range(STEPS):
            pid = start + step * PROGRAMS
            if pid < JOBS:
                group = pid // (W * TILES)
                rank = pid // TILES % W
                x = pid % TILES * B + tl.arange(0, B)
                row = group * GROUP + x // (L * D)
                head = x // D % L
                col = x % D
                gh = rank * L + head
                mask = (row < R) & (x < GROUP * L * D)
                q = tl.load(
                    Q + row * META[4] + gh * META[5] + col,
                    mask,
                    cache_modifier=".cg" if HINT else "",
                )
                k = tl.load(
                    K + row * META[6] + gh * META[7] + col,
                    mask,
                    cache_modifier=".cg" if HINT else "",
                )
                v = tl.load(
                    V + row * META[8] + gh * META[9] + col,
                    mask,
                    cache_modifier=".cg" if HINT else "",
                )
                dst = ((rank * R + row) * L + head) * (3 * D) + col
                tl.store(
                    O + dst, q, mask, cache_modifier=".cs" if STREAM else ""
                )
                tl.store(
                    O + dst + D,
                    k,
                    mask,
                    cache_modifier=".cs" if STREAM else "",
                )
                tl.store(
                    O + dst + 2 * D,
                    v,
                    mask,
                    cache_modifier=".cs" if STREAM else "",
                )


@triton.jit
def _copy_output(
    X,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    start = tl.program_id(0)
    for step in range(STEPS):
        x = (start + step * PROGRAMS) * B + tl.arange(0, B)
        value = tl.load(X + x, x < N)
        tl.store(Y + x, value, x < N)


@triton.jit
def _scalar(Q, K, V, Out, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    Q0: tl.constexpr = META[4]
    Q1: tl.constexpr = META[5]
    K0: tl.constexpr = META[6]
    K1: tl.constexpr = META[7]
    V0: tl.constexpr = META[8]
    V1: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    x = tl.program_id(0) * B + tl.arange(0, B)
    if R * H * D > 0:
        d = x % D
        h = x // D % L
        r = x // (D * L) % R
        rank = x // (D * L * R)
        gh = rank * L + h
        mask = x < R * H * D
        q = tl.load(Q + r * Q0 + gh * Q1 + d, mask)
        k = tl.load(K + r * K0 + gh * K1 + d, mask)
        v = tl.load(V + r * V0 + gh * V1 + d, mask)
        dst = x // D * (3 * D) + d
        tl.store(Out + dst, q, mask)
        tl.store(Out + dst + D, k, mask)
        tl.store(Out + dst + 2 * D, v, mask)


def pack_qkv_destination_major(q, k, v, world_size, out):
    qshape = q.shape
    if len(qshape) != 3 or k.shape != qshape or v.shape != qshape:
        raise ValueError("q, k, v must have matching 3D shapes")
    rows, heads, size = qshape
    if (
        not isinstance(world_size, int)
        or world_size <= 0
        or heads % world_size
    ):
        raise ValueError("world_size must be positive and divide global_heads")
    dtype, device = (q.dtype, q.device)
    if (
        k.dtype != dtype
        or v.dtype != dtype
        or k.device != device
        or (v.device != device)
    ):
        raise ValueError("q, k, v must have matching dtype and device")
    qs, ks, vs = (q.stride(), k.stride(), v.stride())
    if qs[2] != 1 or ks[2] != 1 or vs[2] != 1:
        raise ValueError("q, k, v must have final stride 1")
    local = heads // world_size
    shape = (world_size, rows, local, 3 * size)
    requested = out
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=device)
    elif (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or (not out.is_contiguous())
    ):
        raise ValueError(
            "out must be contiguous with matching shape, dtype and device"
        )
    total = rows * heads * size
    if requested is not None and total:
        element = q.element_size()
        begin = requested.data_ptr()
        end = begin + 3 * total * element
        for tensor, strides in ((q, qs), (k, ks), (v, vs)):
            ptr = tensor.data_ptr()
            if ptr < end and begin < ptr + element * (
                (rows - 1) * strides[0] + (heads - 1) * strides[1] + size
            ):
                out = torch.empty(shape, dtype=dtype, device=device)
                break
    element = q.element_size()
    bytes_per_input = total * element
    strides = (qs[0], qs[1], ks[0], ks[1], vs[0], vs[1])
    if (
        total
        and 3 * total < 2**31
        and 256 * heads * size < 2**31
        and size & (size - 1) == 0
        and qs == ks == vs == (heads * size, size, 1)
    ):
        # Row-batch fast path (see _rowbatch): standard layout, pow2 head
        # size, i32-safe offsets. Row batching keeps the program count low
        # (per-program fixed cost dominates small tiles on GCU). U ladder from
        # a measured sweep (L600, interleaved A/B): per-job cost amortization
        # saturates around U=256-512; U=512 is fastest for rows>=1024
        # (-1.1..-1.7% at rows=4096/16384, neutral at 1024), U=64 stays best
        # for small row counts, num_warps=1 throughout (nw 2/4 regress
        # 2x+). Larger tiles (U=1024) exceed the DSM budget and regress.
        rows_per_program = 64 if rows < 256 else (512 if rows >= 1024 else 256)
        row_blocks = triton.cdiv(rows, rows_per_program)
        jobs = world_size * local * row_blocks
        launched = min(_FIXED_GRID, jobs)
        steps = triton.cdiv(jobs, launched)
        _rowbatch.run(
            q,
            k,
            v,
            out,
            R=rows,
            W=world_size,
            L=local,
            D=size,
            U=rows_per_program,
            RB=row_blocks,
            JOBS=jobs,
            STEPS=steps,
            PROGRAMS=launched,
            num_warps=1,
            grid=(launched,),
            warmup=False,
        )
    elif bytes_per_input < 8388608:
        compact = torch.version.hip is None and qs == ks == vs == (
            heads * size,
            size,
            1,
        )
        block = 256 if compact else 1024
        warps = 2 if compact else 4
        _scalar.run(
            q,
            k,
            v,
            out,
            (rows, heads, size, local) + strides + (block,),
            num_warps=warps,
            grid=(max(1, triton.cdiv(total, block)),),
            warmup=False,
        )
    elif torch.version.hip is not None and bytes_per_input < 33554432:
        group = 32
        block = 512
        programs = max(
            1,
            3
            * world_size
            * triton.cdiv(rows, group)
            * triton.cdiv(group * local * size, block),
        )
        _split.run(
            q,
            k,
            v,
            out,
            (rows, heads, size, local) + strides + (block,),
            group,
            True,
            num_warps=4,
            grid=(programs,),
            warmup=False,
        )
    else:
        group = 32 if bytes_per_input < 67108864 else 128
        block = 512
        hip = torch.version.hip is not None
        hint = (
            hip
            and bytes_per_input >= 33554432
            or (not hip and bytes_per_input < 33554432)
        )
        stream = hint and (bytes_per_input >= 67108864 or not hip)
        programs = max(
            1,
            world_size
            * triton.cdiv(rows, group)
            * triton.cdiv(group * local * size, block),
        )
        launched = min(programs, _GRID_X_MAX)
        steps = triton.cdiv(programs, launched)
        _grouped.run(
            q,
            k,
            v,
            out,
            (rows, heads, size, local) + strides + (block,),
            group,
            hint,
            stream,
            steps,
            launched,
            num_warps=4,
            grid=(launched,),
            warmup=False,
        )
    if requested is not None and out is not requested:
        jobs = max(1, triton.cdiv(3 * total, 1024))
        launched = min(jobs, _GRID_X_MAX)
        steps = triton.cdiv(jobs, launched)
        _copy_output.run(
            out,
            requested,
            3 * total,
            1024,
            steps,
            launched,
            grid=(launched,),
            warmup=False,
        )
        return requested
    return out


__all__ = ["pack_qkv_destination_major"]
