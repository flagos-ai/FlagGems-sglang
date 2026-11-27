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

"""Verbatim SGLang kpool assemble, compress and cache-write implementation.

Source: sglang/srt/layers/attention/nsa/kpool/kernels.py.
Source SHA256: 18626aabd83127fe452bbe16a31e5207cf875a54d776717cc3f497e944175913.
"""

import json
import os
import threading
import time

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128

_KPOOL_CALL_LOG_LOCK = threading.Lock()
_KPOOL_CALL_LOG_COUNTS: dict[str, int] = {}


def _kpool_call_log_value(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": list(value.stride()),
            "is_contiguous": value.is_contiguous(),
            "requires_grad": value.requires_grad,
        }

    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": value_type, "value": value}

    result = {"type": value_type, "value": repr(value)}
    for attr in ("index_kpool", "slots_per_page"):
        if hasattr(value, attr):
            attr_value = getattr(value, attr)
            if isinstance(attr_value, (bool, int, float, str)):
                result[attr] = attr_value
            else:
                result[attr] = repr(attr_value)
    return result


def _log_kpool_wrapper_call(function_name: str, **kwargs) -> None:
    log_path = os.environ.get("SGLANG_KPOOL_CALL_LOG")
    if not log_path:
        return

    try:
        max_calls = int(os.environ.get("SGLANG_KPOOL_CALL_LOG_MAX", "512"))
    except ValueError:
        max_calls = 512

    with _KPOOL_CALL_LOG_LOCK:
        call_index = _KPOOL_CALL_LOG_COUNTS.get(function_name, 0)
        if call_index > max_calls:
            return
        _KPOOL_CALL_LOG_COUNTS[function_name] = call_index + 1

        if call_index == max_calls:
            record = {
                "event": "limit_reached",
                "function": function_name,
                "pid": os.getpid(),
                "max_calls": max_calls,
            }
        else:
            try:
                is_capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception as exc:
                is_capturing = f"unavailable:{type(exc).__name__}"
            record = {
                "event": "call",
                "timestamp_ns": time.time_ns(),
                "function": function_name,
                "call_index": call_index,
                "pid": os.getpid(),
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "is_cuda_graph_capturing": is_capturing,
                "args": {
                    name: _kpool_call_log_value(value)
                    for name, value in kwargs.items()
                },
            }

        line = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode()
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _round_to_nearest_int(x):
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _hadamard_quantize_int8(acc, denom, ROUND_SCALE: tl.constexpr):
    """Normalize -> bf16 round-trip -> Hadamard rotate -> bf16 round-trip
    -> int8 quantize. Returns (quantized, scale)."""
    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    int8_max_inv = 1.0 / 127.0
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * int8_max_inv)))
    else:
        scale = absmax * int8_max_inv

    quantized = tl.minimum(tl.maximum(x / scale, -127.0), 127.0)
    quantized = _round_to_nearest_int(quantized).to(tl.int8)
    return quantized, scale


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_kernel(
    buf_i8_ptr,
    buf_fp32_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    tail_logical_base_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr = 64,
):
    """Per-pool program. Slot src dispatch:
    if slot < n_from_tail[row]:
        src = tail[req_pool_idx[row], (tail_logical_base[row] + slot) % TAIL_SIZE]
    else:
        chunk_off = chunk_src_start[row] + (slot - n_from_tail[row])
        src = chunk[chunk_off]
    """
    row = tl.program_id(0)
    # CP rows masked out have no result to compute; skip the full
    # POOL_SIZE-load + Hadamard + fp8 quant pipeline (the previous
    # implementation paid them and only masked the final store).
    if HAS_WRITE_MASK:
        if not tl.load(write_mask_ptr + row):
            return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    n_tail = tl.load(n_from_tail_ptr + row)
    req = tl.load(req_pool_idx_ptr + row)
    chunk_src = tl.load(chunk_src_start_ptr + row)
    tail_base = tl.load(tail_logical_base_ptr + row)

    # Online softmax over POOL_SIZE slots; (max, denom, acc) rescaled
    # in place to halve the load traffic vs a two-pass version.
    #
    # `from_tail = slot < n_tail` is a scalar runtime bool (n_tail is a
    # per-row scalar). Using a real `if/else` here -- instead of the
    # previous `tl.where(from_tail, load_tail, load_chunk)` -- means
    # only the active branch's loads are emitted. The dead pointer
    # arithmetic (e.g. ``tail_ptr + chunk_off``) is never computed, so
    # we cannot read into unmapped memory even when max_running_requests
    # is small.
    m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        if slot < n_tail:
            start = (tail_base % TAIL_SIZE).to(tl.int64)
            phys = (start + slot) % TAIL_SIZE
            off = req * tail_stride_0 + phys * tail_stride_1 + offs
            score = tl.load(tail_score_ptr + off, mask=mask, other=0.0).to(
                tl.float32
            )
            k = tl.load(tail_k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        else:
            off = (chunk_src + (slot - n_tail)) * chunk_stride_0 + offs
            score = tl.load(chunk_score_ptr + off, mask=mask, other=0.0).to(
                tl.float32
            )
            k = tl.load(chunk_k_ptr + off, mask=mask, other=0.0).to(tl.float32)

        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0
        ).to(tl.float32)

        new_m = tl.maximum(m, score)
        rescale = tl.exp(m - new_m)
        prob = tl.exp(score - new_m)
        denom = denom * rescale + prob
        acc = acc * rescale + k * prob
        m = new_m

    quantized, scale = _hadamard_quantize_int8(acc, denom, ROUND_SCALE)

    loc = tl.load(loc_ptr + row)
    loc_page_index = loc // SLOTS_PER_PAGE
    loc_token_offset_in_page = loc % SLOTS_PER_PAGE
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * HEAD_DIM
        + offs
    )
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_i8_ptr + out_k_offsets, quantized, mask=mask)
    tl.store(buf_fp32_ptr + out_s_offset, scale)


def kpool_assemble_softmax_rotate_write_cache(
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
    round_scale: bool = False,
) -> None:
    """Fused gather + softmax + Hadamard rotate + int8 quant + cache write.

    For each output pool row r, slots are gathered from tail or chunk
    based on n_from_tail[r] (see _kpool_assemble_softmax_rotate_write_cache_kernel).
    ``tail_logical_base[r]`` is the logical position of the in-progress pool's
    first slot (used to ring-address the saved tail prefix).
    """
    _log_kpool_wrapper_call(
        "kpool_assemble_softmax_rotate_write_cache",
        pool=pool,
        buf=buf,
        chunk_k=chunk_k,
        chunk_score=chunk_score,
        tail_k=tail_k,
        tail_score=tail_score,
        req_pool_idx=req_pool_idx,
        n_from_tail=n_from_tail,
        chunk_src_start=chunk_src_start,
        tail_logical_base=tail_logical_base,
        ape=ape,
        loc=loc,
        write_mask=write_mask,
        round_scale=round_scale,
    )
    pool_size = pool.index_kpool
    n_pools = req_pool_idx.shape[0]

    if n_pools == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True

    buf_i8 = buf.view(torch.int8)
    buf_fp32 = buf.view(torch.float32)
    slots_per_page = pool.slots_per_page

    _kpool_assemble_softmax_rotate_write_cache_kernel[(n_pools,)](
        buf_i8,
        buf_fp32,
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
        write_mask,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=slots_per_page,
    )


__all__ = ["kpool_assemble_softmax_rotate_write_cache"]
