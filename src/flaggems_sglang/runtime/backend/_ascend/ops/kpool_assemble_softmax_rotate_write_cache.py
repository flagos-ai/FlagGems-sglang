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

"""
v1 uses TLE for cached plan reads and the final two Hadamard stages.
"""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

_selected_version = os.environ.get("GET_FLAGGEMS_VERSION", "v0")
if _selected_version not in ("v0", "v1"):
    raise ValueError(
        "GET_FLAGGEMS_VERSION must be 'v0' or 'v1', got "
        f"{_selected_version!r}"
    )
if _selected_version == "v1":
    try:
        from triton.experimental.tle.language import dsa
    except ImportError as error:
        raise ImportError(
            "GET_FLAGGEMS_VERSION=v1 requires Triton TLE "
            "with Ascend DSA support"
        ) from error


@triton.jit
def _hadamard128(x, high_stages: tl.constexpr):
    cols = tl.arange(0, 128)
    gather_stages: tl.constexpr = 7 if high_stages is None else 5
    for stage in tl.static_range(gather_stages):
        stride = 1 << stage
        other = tl.gather(x, cols ^ stride, axis=0)
        sign = tl.where((cols & stride) == 0, 1.0, -1.0)
        x = tl.fma(x, sign, other)
    if high_stages is not None:
        x = high_stages(x)
    return x * 0.08838834764831845


@triton.jit
def _round_to_nearest_int(x):
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _write_pool(
    key_cache,
    scale_cache,
    page_slots,
    acc,
    denom,
    loc,
    cols,
    high_stages: tl.constexpr,
    PAGE_BYTES: tl.constexpr = None,
    ROUND_SCALE: tl.constexpr = True,
):
    pooled = (acc / denom).to(tl.bfloat16).to(tl.float32)
    rotated = _hadamard128(pooled, high_stages).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(rotated), 0), 1e-4)
    scale = absmax * (1.0 / 127.0)
    if ROUND_SCALE:
        # Carry a nonzero fraction into the exponent for power-of-two rounding.
        bits = scale.to(tl.uint32, bitcast=True)
        scale = ((bits + 0x7FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
    quant = tl.minimum(tl.maximum(rotated / scale, -127.0), 127.0)
    encoded = _round_to_nearest_int(quant).to(tl.int8)
    page, slot = loc // page_slots, loc % page_slots
    # Each page contains all 128-byte keys, then one FP32 scale per slot.
    if PAGE_BYTES is None:
        page_start = page * page_slots
        key_base = page_start * 132
        scale_base = page_start * 33
    else:
        key_base = page * PAGE_BYTES
        scale_base = page * (PAGE_BYTES // 4)
    tl.store(key_cache + key_base + slot * 128 + cols, encoded)
    scale_offset = scale_base + page_slots * 32 + slot
    tl.store(scale_cache + scale_offset, scale)


@triton.jit
def _gather_i32(values, pick):
    # Ascend gather takes FP32 lanes. Bitcast both ways to preserve all bits.
    bits = tl.gather(values.to(tl.float32, bitcast=True), pick, 0)
    return tl.sum(bits.to(tl.int32, bitcast=True), 0)


@triton.jit
def _cached_element_v0(values, index):
    pick = tl.full((1,), index, tl.int32)
    if values.dtype == tl.int32:
        return _gather_i32(values, pick)
    elif values.dtype == tl.int64:
        low, high = values.to(tl.int32), (values >> 32).to(tl.int32)
        low, high = _gather_i32(low, pick), _gather_i32(high, pick)
        return (high.to(tl.int64) << 32) | low.to(tl.uint32).to(tl.int64)
    else:
        return tl.sum(
            tl.gather(values.to(tl.float32), pick, 0).to(tl.int32), 0
        )


if _selected_version == "v1":

    @triton.jit
    def _hadamard_slice_stage(x, width: tl.constexpr):
        result = x
        for block in tl.static_range(128 // (2 * width)):
            start = 2 * width * block
            low = dsa.extract_slice(x, (start,), (width,), (1,))
            high = dsa.extract_slice(x, (start + width,), (width,), (1,))
            result = dsa.insert_slice(
                result, low + high, (start,), (width,), (1,)
            )
            result = dsa.insert_slice(
                result, low - high, (start + width,), (width,), (1,)
            )
        return result

    @triton.jit
    def _hadamard_high_stages(x):
        return _hadamard_slice_stage(_hadamard_slice_stage(x, 32), 64)

    @triton.jit
    def _cached_element_v1(values, index):
        return dsa.extract_element(values, (index,))

else:
    _cached_element_v1 = None
    _hadamard_high_stages = None


@triton.jit
def _online_pool(
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    ape_ptr,
    chunk_stride,
    req_stride,
    tail_stride,
    ape_stride,
    cols,
    req,
    n_tail,
    chunk_src,
    tail_base,
    pool_size,
    tail_size,
    FULL_WRAP: tl.constexpr,
):
    maximum = tl.full((128,), -float("inf"), tl.float32)
    acc = tl.full((128,), 0.0, tl.float32)
    denom = tl.full((128,), 0.0, tl.float32)
    for slot in range(pool_size):
        if slot < n_tail:
            start = (tail_base % tail_size).to(tl.int64)
            physical = start + slot
            if FULL_WRAP:
                physical %= tail_size
            else:
                physical = tl.where(
                    physical >= tail_size, physical - tail_size, physical
                )
            physical = physical.to(req.dtype)
            offset = req * req_stride + physical * tail_stride + cols
            key = tl.load(tail_k + offset).to(tl.float32)
            score = tl.load(tail_score + offset).to(tl.float32)
        else:
            offset = (chunk_src + slot - n_tail) * chunk_stride + cols
            key = tl.load(chunk_k + offset).to(tl.float32)
            score = tl.load(chunk_score + offset).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride + cols).to(tl.float32)
        new_maximum = tl.maximum(maximum, score)
        rescale = tl.exp(maximum - new_maximum)
        weight = tl.exp(score - new_maximum)
        acc = acc * rescale + key * weight
        denom = denom * rescale + weight
        maximum = new_maximum
    return acc, denom


@triton.jit
def _load_plan_element(field, row):
    return tl.load(field + row)


@triton.jit
def _dense_pool(
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    ape,
    chunk_stride,
    req_stride,
    tail_stride,
    cols,
    req_plan,
    n_tail_plan,
    src_plan,
    base_plan,
    row,
    tail_size,
    K: tl.constexpr,
    read_plan: tl.constexpr,
    index_type: tl.constexpr,
):
    slots = tl.arange(0, K)[:, None]
    n_tail = read_plan(n_tail_plan, row)
    if n_tail == 0:
        # Full chunk pools use a contiguous tile with no slot mask.
        chunk_src = read_plan(src_plan, row).to(index_type)
        offset = (chunk_src + slots) * chunk_stride + cols[None, :]
        keys = tl.load(chunk_k + offset).to(tl.float32)
        scores = tl.load(chunk_score + offset).to(tl.float32)
    else:
        req = read_plan(req_plan, row).to(index_type)
        tail_base = read_plan(base_plan, row)
        physical = (tail_base % tail_size).to(req.dtype) + slots
        physical = tl.where(
            physical >= tail_size, physical - tail_size, physical
        )
        offset = req * req_stride + physical * tail_stride + cols[None, :]
        from_tail = slots < n_tail
        keys = tl.load(tail_k + offset, from_tail, other=0).to(tl.float32)
        scores = tl.load(tail_score + offset, from_tail, other=0).to(
            tl.float32
        )
        # A closed extend pool always includes at least one chunk token.
        chunk_src = read_plan(src_plan, row).to(index_type)
        offset = (chunk_src + slots - n_tail) * chunk_stride + cols[None, :]
        keys += tl.load(chunk_k + offset, ~from_tail, other=0).to(tl.float32)
        scores += tl.load(chunk_score + offset, ~from_tail, other=0).to(
            tl.float32
        )
    logits = scores + ape
    weights = tl.exp(logits - tl.max(logits, 0)[None, :])
    return tl.sum(keys * weights, 0), tl.sum(weights, 0)


@triton.jit
def _load_ape(ape_ptr, stride, cols, K: tl.constexpr):
    slots = tl.arange(0, K)[:, None]
    return tl.load(ape_ptr + slots * stride + cols[None, :]).to(tl.float32)


@triton.jit(
    do_not_specialize=[
        "buf_i8",
        "buf_f32",
        "chunk_k",
        "chunk_score",
        "tail_k",
        "tail_score",
        "req_ptr",
        "n_tail_ptr",
        "chunk_src_ptr",
        "tail_base_ptr",
        "ape_ptr",
        "loc_ptr",
        "mask_ptr",
        "count",
        "chunk_stride",
        "req_stride",
        "tail_stride",
        "ape_stride",
        "page_slots",
        "pool_size",
        "tail_size",
        "has_mask",
    ]
)
def _assemble_kernel(
    buf_i8,
    buf_f32,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_ptr,
    n_tail_ptr,
    chunk_src_ptr,
    tail_base_ptr,
    ape_ptr,
    loc_ptr,
    mask_ptr,
    count,
    chunk_stride,
    req_stride,
    tail_stride,
    ape_stride,
    page_slots,
    pool_size,
    tail_size,
    has_mask,
    K: tl.constexpr,
    PLAN: tl.constexpr,
    INDEX32: tl.constexpr,
    USE_TLE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)
    index_type: tl.constexpr = tl.int32 if INDEX32 else tl.int64
    if USE_TLE:
        read_cached: tl.constexpr = _cached_element_v1
        high_stages: tl.constexpr = _hadamard_high_stages
    else:
        read_cached: tl.constexpr = _cached_element_v0
        high_stages: tl.constexpr = None
    sources = (chunk_k, chunk_score, tail_k, tail_score)
    strides = (chunk_stride, req_stride, tail_stride)
    cache = (buf_i8, buf_f32, page_slots)
    if PLAN == "direct":
        if has_mask:
            live = tl.load(mask_ptr + pid).to(tl.int1)
        else:
            live = True
        # Guard plan loads as masked rows may contain invalid indices.
        if live:
            if K == 0:
                req = tl.load(req_ptr + pid).to(index_type)
                n_tail = tl.load(n_tail_ptr + pid)
                src = tl.load(chunk_src_ptr + pid).to(index_type)
                base = tl.load(tail_base_ptr + pid)
                acc, denom = _online_pool(
                    *sources,
                    ape_ptr,
                    *strides,
                    ape_stride,
                    cols,
                    req,
                    n_tail,
                    src,
                    base,
                    pool_size,
                    tail_size,
                    PAGE_BYTES is not None,
                )
            else:
                ape = _load_ape(ape_ptr, ape_stride, cols, K)
                plan = (req_ptr, n_tail_ptr, chunk_src_ptr, tail_base_ptr)
                acc, denom = _dense_pool(
                    *sources,
                    ape,
                    *strides,
                    cols,
                    *plan,
                    pid,
                    tail_size,
                    K,
                    _load_plan_element,
                    index_type,
                )
            dst = tl.load(loc_ptr + pid).to(index_type)
            _write_pool(
                *cache,
                acc,
                denom,
                dst,
                cols,
                high_stages,
                PAGE_BYTES,
                ROUND_SCALE,
            )
    else:
        tile_size: tl.constexpr = 64
        grid = tl.num_programs(0)
        per_core = count // grid
        remainder = count % grid
        start = pid * per_core + tl.minimum(pid, remainder)
        end = start + per_core + (pid < remainder)
        ape = _load_ape(ape_ptr, ape_stride, cols, K)
        for first in range(start, end, tile_size):
            rows = first + tl.arange(0, tile_size)
            live = rows < end
            if PLAN == "cached_masked":
                live &= tl.load(mask_ptr + rows, live, other=0).to(tl.int1)
            if tl.sum(live.to(tl.int32), 0) > 0:
                req = tl.load(req_ptr + rows, live, other=0).to(index_type)
                n_tail = tl.load(n_tail_ptr + rows, live, other=0)
                src = tl.load(chunk_src_ptr + rows, live, other=0).to(
                    index_type
                )
                base = tl.load(tail_base_ptr + rows, live, other=0)
                loc = tl.load(loc_ptr + rows, live, other=0).to(index_type)
                plan = (req, n_tail, src, base)
                active = live.to(tl.int32)
                for local in range(tl.minimum(tile_size, end - first)):
                    if PLAN == "cached_masked":
                        pool_active = read_cached(active, local) != 0
                    else:
                        pool_active = True
                    if pool_active:
                        acc, denom = _dense_pool(
                            *sources,
                            ape,
                            *strides,
                            cols,
                            *plan,
                            local,
                            tail_size,
                            K,
                            read_cached,
                            index_type,
                        )
                        dst = read_cached(loc, local)
                        _write_pool(*cache, acc, denom, dst, cols, high_stages)


@lru_cache(maxsize=None)
def _core_count(device):
    properties = triton.runtime.driver.active.utils.get_device_properties(
        device
    )
    return properties["num_vectorcore"]


def _make_entrypoint(name, *, use_tle):
    def assemble(
        pool,
        buf: torch.Tensor,
        chunk_k: torch.Tensor,
        chunk_score: torch.Tensor,
        tail_k: torch.Tensor,
        tail_score: torch.Tensor,
        req_pool_idx: torch.Tensor,
        n_from_tail: torch.Tensor,
        chunk_src_start: torch.Tensor,
        tail_logical_base: torch.Tensor,
        ape: torch.Tensor,
        loc: torch.Tensor,
        write_mask: torch.Tensor | None = None,
        round_scale: bool = True,
    ) -> None:
        if use_tle and _selected_version != "v1":
            raise RuntimeError(
                "Set GET_FLAGGEMS_VERSION=v1 before importing this module"
            )
        count = req_pool_idx.numel()
        if not count:
            return

        chunk_k = chunk_k.contiguous()
        chunk_score = chunk_score.contiguous()
        loc = loc.contiguous()
        ape = ape.contiguous()

        pool_size, tail_size = pool.index_kpool, tail_k.shape[1]
        page_slots = pool.slots_per_page
        page_bytes = buf.shape[1]
        # The fast kernel uses packed 64-token pages, a single tail wrap,
        # and power-of-two scales. Other inputs use the general online path.
        use_fast_path = (
            round_scale
            and pool_size > 1
            and 64 % pool_size == 0
            and tail_size >= pool_size
            and page_slots == 64 // pool_size
            and page_bytes == page_slots * 132
        )
        dense = use_fast_path and pool_size in (2, 4, 8) and tail_size < 2**30
        block_k = pool_size if dense else 0
        cores = _core_count(chunk_k.device.index or 0)
        cache_plan = dense and count > cores
        grid = cores if cache_plan else count

        tail_span = (
            (tail_k.shape[0] - 1) * tail_k.stride(0)
            + (tail_size - 1) * tail_k.stride(1)
            + 128
        )
        # KV offsets can address FP32 elements; the cache uses byte offsets.
        index32 = (
            max(tail_span, chunk_k.numel()) < 2**29 and buf.numel() < 2**31
        )

        has_mask = write_mask is not None
        plan = "direct"
        if cache_plan:
            plan = "cached_masked" if has_mask else "cached"
        # A bool placeholder keeps the kernel signature stable without a mask.
        mask = (
            req_pool_idx.view(torch.bool)
            if write_mask is None
            else write_mask.contiguous()
        )
        _assemble_kernel[(grid,)](
            buf.view(torch.int8),
            buf.view(torch.float32),
            chunk_k,
            chunk_score,
            tail_k,
            tail_score,
            req_pool_idx,
            n_from_tail,
            chunk_src_start,
            tail_logical_base,
            ape,
            loc,
            mask,
            count,
            chunk_k.stride(0),
            tail_k.stride(0),
            tail_k.stride(1),
            ape.stride(0),
            page_slots,
            pool_size,
            tail_size,
            int(has_mask),
            block_k,
            plan,
            index32,
            use_tle,
            None if use_fast_path else page_bytes,
            round_scale,
        )

    assemble.__name__ = assemble.__qualname__ = name
    return assemble


_kpool_assemble_softmax_rotate_write_cache_v0 = _make_entrypoint(
    "_kpool_assemble_softmax_rotate_write_cache_v0", use_tle=False
)
kpool_assemble_softmax_rotate_write_cache_v1 = _make_entrypoint(
    "kpool_assemble_softmax_rotate_write_cache_v1",
    use_tle=True,
)

kpool_assemble_softmax_rotate_write_cache = (
    _kpool_assemble_softmax_rotate_write_cache_v0
    if _selected_version == "v0"
    else kpool_assemble_softmax_rotate_write_cache_v1
)

__all__ = ["kpool_assemble_softmax_rotate_write_cache"]
