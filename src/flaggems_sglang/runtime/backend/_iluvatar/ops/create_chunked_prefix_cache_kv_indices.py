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

"""create_chunked_prefix_cache_kv_indices Triton kernel (Batch 7, Task 95).

EXACT CONTRACT (platform statement, VERIFIED 2026-09-24)
    def reference(req_to_token, req_pool_indices, chunk_start_idx,
                  chunk_seq_lens, chunk_cu_seq_lens, chunk_kv_indices):
        out = chunk_kv_indices.clone()
        for i in range(req_pool_indices.shape[0]):
            beg = int(chunk_cu_seq_lens[i]);  n = int(chunk_seq_lens[i])
            start = int(chunk_start_idx[i]);  pool = int(req_pool_indices[i])
            out[beg : beg + n] = req_to_token[pool, start : start + n]
        return out
    Per request i, copy the token window
    req_to_token[pool_i, start_i : start_i + n_i] into the flat CSR stream
    chunk_kv_indices[cu_i : cu_i + n_i].  In-place write into
    chunk_kv_indices and return it (reference clones; the values the
    runner compares are identical).  Correctness: EXACT integer.

Design notes
------------
* Grid = (batch,) -- one program per request, 512-wide block loop over
  the chunk window (the upstream sglang kernel shape, known to compile
  and run on every backend; the loop holds NO loop-carried scalar
  accumulator, only a range index).
* req_pool_index stays int32: pool_index * max_context would need
  > 2^31 to overflow -- far beyond testable tables.  No int64 anywhere
  (enflame TLE cannot lower it).
* Masked vector loads/stores only (kunlunxin store-mask rules); the
  offsets math is int32 vector + scalar.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ccpk_rows(
    req_to_token_ptr, req_pool_indices_ptr, chunk_start_idx_ptr,
    chunk_seq_lens_ptr, chunk_cu_seq_lens_ptr, chunk_kv_indices_ptr,
    req_to_token_stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    out_base = tl.load(chunk_cu_seq_lens_ptr + pid)
    chunk_start = tl.load(chunk_start_idx_ptr + pid)
    chunk_len = tl.load(chunk_seq_lens_ptr + pid)

    src_row = req_to_token_ptr + req_pool_index * req_to_token_stride + chunk_start
    num_loop = tl.cdiv(chunk_len, BLOCK)
    for i in range(num_loop):
        cols = i * BLOCK + tl.arange(0, BLOCK)
        m = cols < chunk_len
        data = tl.load(src_row + cols, mask=m)
        tl.store(chunk_kv_indices_ptr + out_base + cols, data, mask=m)


def _detect_vendor(device):
    # Platform upload precheck forbids importing os -- vendor routing keys
    # on device.type only (gcu->enflame, npu->ascend, xpu->kunlunxin);
    # CUDA-class devices take the default order chain.
    return {"gcu": "enflame", "npu": "ascend", "xpu": "kunlunxin"}.get(
        device.type, device.type
    )


def create_chunked_prefix_cache_kv_indices(*args, **kwargs):
    """In-place CSR gather; returns chunk_kv_indices.

    Argument conventions handled (never raises on convention): the platform
    calls with NAMED arguments matching the reference signature -- those
    names win.  Positional fallback follows the contract order
    (req_to_token, req_pool_indices, chunk_start_idx, chunk_seq_lens,
    chunk_cu_seq_lens, chunk_kv_indices); no width heuristics (cu and
    chunk_kv_indices can share the same max width, so max() would misfire).
    """
    named = {}
    pos = []
    for name in ("req_to_token", "req_pool_indices", "chunk_start_idx",
                 "chunk_seq_lens", "chunk_cu_seq_lens", "chunk_kv_indices"):
        v = kwargs.get(name)
        if isinstance(v, torch.Tensor):
            named[name] = v
    pos = [t for t in args if isinstance(t, torch.Tensor)]
    order = ("req_to_token", "req_pool_indices", "chunk_start_idx",
             "chunk_seq_lens", "chunk_cu_seq_lens", "chunk_kv_indices")
    tensors = [named.get(n) for n in order]
    if len(pos) == 6 and any(t is None for t in tensors):
        tensors = pos
    if any(t is None for t in tensors):
        raise RuntimeError(
            f"create_chunked_prefix_cache_kv_indices: could not bind all 6 "
            f"tensors (named={sorted(named.keys())}, n_positional={len(pos)})"
        )
    (req_to_token, req_pool_indices, chunk_start_idx, chunk_seq_lens,
     chunk_cu_seq_lens, chunk_kv_indices) = tensors

    bs = req_pool_indices.shape[0]
    if bs == 0:
        return chunk_kv_indices

    def _i32(t):
        if t.dtype == torch.int64:
            return t.to(torch.int32).contiguous()
        return t.contiguous()

    req_to_token_c = req_to_token.contiguous()
    pool_c = _i32(req_pool_indices)
    start_c = _i32(chunk_start_idx)
    lens_c = _i32(chunk_seq_lens)
    cu_c = _i32(chunk_cu_seq_lens)
    out_c = chunk_kv_indices if chunk_kv_indices.is_contiguous() \
        else chunk_kv_indices.contiguous()

    vendor = _detect_vendor(req_to_token.device)
    errors = []
    # Round-2 (2026-09-25): the reference is a per-request Python loop, so
    # every Triton variant is already 20-106x; widening BLOCK cuts the inner
    # loop iterations (less loop/launch overhead) -- the cheapest lever to
    # close the 42.67x -> 49.11x leader gap.  ascend/huawei keep a 1024 head
    # (verified 24.67x) then widen.
    if vendor in ("enflame", "gcu"):
        orders = ((1024, 4), (2048, 4), (512, 4), (4096, 4), (256, 4))
    elif vendor in ("hygon", "ascend", "npu"):
        # r4 (2026-09-25): keep the r3b 2048-head win for 海/华 (50.79/31.11).
        orders = ((2048, 4), (1024, 4), (4096, 4), (512, 4), (256, 4))
    elif vendor == "iluvatar":
        # r4: 沐 recover to the r2 medium config (25.29 best) -- r3b wide hurt it.
        orders = ((2048, 4), (4096, 4), (1024, 4), (512, 4), (256, 4))
    else:
        # metax/kunlunxin/国A/国B: r3b wide wins (天139.84/昆39.38/国A36.97/国B36.28).
        orders = ((4096, 8), (2048, 4), (1024, 8), (8192, 8), (512, 4))

    for blk, nw in orders:
        try:
            _ccpk_rows[(bs,)](
                req_to_token_c, pool_c, start_c, lens_c, cu_c, out_c,
                req_to_token_c.stride(0),
                BLOCK=blk, num_warps=nw,
            )
            return out_c
        except Exception as e:  # noqa: BLE001
            errors.append(f"b{blk}_w{nw}: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"create_chunked_prefix_cache_kv_indices all variants failed "
        f"(device={req_to_token.device.type}, bs={bs}) | " + " || ".join(errors)
    )


__all__ = ["create_chunked_prefix_cache_kv_indices"]
