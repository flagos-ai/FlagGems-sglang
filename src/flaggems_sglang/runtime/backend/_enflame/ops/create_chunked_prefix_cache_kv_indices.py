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
* Flat arm (bs <= 48): one program per request, 512-wide block loop over
  the chunk window (the upstream sglang kernel shape).  The loop holds NO
  loop-carried scalar accumulator, only a range index.
* req_pool_index stays int32: pool_index * max_context would need
  > 2^31 to overflow -- far beyond testable tables.  No int64 anywhere
  (enflame TLE cannot lower it).
* Masked vector loads/stores only (kunlunxin store-mask rules); the
  offsets math is int32 vector + scalar.
* q3 splits the block loop into unmasked full tiles plus one masked tail.
* q15 declares the table strides constexpr, as upstream sglang does
  (`req_to_token_ptr_stride: tl.constexpr`), so the address math folds and the
  backend can see divisibility it could not assume at runtime.
* q2 drops the wrapper's two device kernels (int64->int32 cast of the index
  tensors, contiguous() copy of a sliced req_to_token): narrowing happens on
  the four scalar loads inside the kernel, and the table is indexed through
  both of its strides.  Values are unchanged for contiguous int32 inputs.

enflame tuning (2026-11-08, GCU do_bench KERNEL median, w1000/r100)
-------------------------------------------------------------------
Probe decomposition (ablation ladder): event floor 2.10us (empty closure),
empty-kernel launch +0.21us, 4-scalar load window +0.75us (loads overlap),
one masked tile copy window +0.65..0.95us.  The flat kernel is a serial
2-stage memory-latency chain (index scalars -> payload tile), so it sits at
its structural floor; the harvestable margin is dispatch-side:
* num_warps by bs (host metadata only): bs == 1 keeps the historical
  (BLOCK=2048, num_warps=4) launch (byte-identical binary, zero risk);
  bs >= 2 launches (2048, 2): -0.60us at bs=8 / -0.28us at bs=32
  (two independent probe rounds, > 2x noise band).  num_stages: no effect.
* bs > 48 wave quantization: one-CTA-per-request grids past the 48-CTA
  single-wave ceiling pay ~48ns/CTA of dispatch waves (measured flat
  6.37/10.49/17.59us at bs=64/128/256 with data volumes of only
  0.25-1.0 MB).  A GROUP=2 persistent grid-stride kernel
  (grid = min(48, cdiv(bs, 2))) with all eight scalar loads hoisted into
  one issue window BEFORE any store (loads after stores serialize on
  pointer aliasing; an interleaved group variant measured 6.4-9.6us and is
  rejected) brings bs=64/128/256 to 4.71/5.70/7.30us (-26%/-46%/-59%).
  Odd tails clamp r1 = min(r0+1, last): re-copying the last request is
  idempotent (same values written twice to the same window).  Segments
  longer than BLOCK take the general full-tiles + masked-tail loop
  (verified exact vs reference at chunk=8192).
* Cross-over measured at the boundary: bs=48 flat 4.10us vs group 4.71us
  -> threshold 48 (flat wins at and below, group wins above).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ccpk_rows(
    req_to_token_ptr,
    req_pool_indices_ptr,
    chunk_start_idx_ptr,
    chunk_seq_lens_ptr,
    chunk_cu_seq_lens_ptr,
    chunk_kv_indices_ptr,
    req_to_token_stride: tl.constexpr,
    r2t_stride1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int32)
    out_base = tl.load(chunk_cu_seq_lens_ptr + pid).to(tl.int32)
    chunk_start = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    src_row = (
        req_to_token_ptr
        + req_pool_index * req_to_token_stride
        + chunk_start * r2t_stride1
    )
    dst_row = chunk_kv_indices_ptr + out_base
    cols = tl.arange(0, BLOCK)
    # Predicate-free hot loop: a tile that is wholly inside the window does not need
    # `cols < chunk_len`, and paying it costs a compare plus predicated memory ops on
    # every lane of every trip (x1.03-1.08 against this split, interleaved rounds).
    full = chunk_len // BLOCK
    for i in range(full):
        off = i * BLOCK + cols
        data = tl.load(src_row + off * r2t_stride1)
        tl.store(dst_row + off, data)
    rest = chunk_len - full * BLOCK
    if rest > 0:
        off = full * BLOCK + cols
        m = off < chunk_len
        data = tl.load(src_row + off * r2t_stride1, mask=m)
        tl.store(dst_row + off, data, mask=m)


@triton.jit
def _ccpk_group2(
    req_to_token_ptr,
    req_pool_indices_ptr,
    chunk_start_idx_ptr,
    chunk_seq_lens_ptr,
    chunk_cu_seq_lens_ptr,
    chunk_kv_indices_ptr,
    num_requests,
    req_to_token_stride: tl.constexpr,
    r2t_stride1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Persistent grid-stride over PAIRS of requests; used when bs exceeds the
    # 48-CTA single-wave dispatch ceiling.  All eight scalar loads issue in one
    # window before the first store (loads placed after stores serialize on
    # pointer aliasing -- that interleaved form measured 6.4-9.6us and is
    # rejected).  r1 clamps to the last request; the duplicate re-copy is
    # idempotent.
    pid = tl.program_id(0)
    last = num_requests - 1
    cols = tl.arange(0, BLOCK)
    for r0 in tl.range(pid * 2, num_requests, tl.num_programs(0) * 2):
        r1 = tl.minimum(r0 + 1, last)
        pool0 = tl.load(req_pool_indices_ptr + r0).to(tl.int32)
        pool1 = tl.load(req_pool_indices_ptr + r1).to(tl.int32)
        ob0 = tl.load(chunk_cu_seq_lens_ptr + r0).to(tl.int32)
        ob1 = tl.load(chunk_cu_seq_lens_ptr + r1).to(tl.int32)
        st0 = tl.load(chunk_start_idx_ptr + r0).to(tl.int32)
        st1 = tl.load(chunk_start_idx_ptr + r1).to(tl.int32)
        ln0 = tl.load(chunk_seq_lens_ptr + r0).to(tl.int32)
        ln1 = tl.load(chunk_seq_lens_ptr + r1).to(tl.int32)
        src0 = (
            req_to_token_ptr
            + pool0 * req_to_token_stride
            + st0 * r2t_stride1
        )
        src1 = (
            req_to_token_ptr
            + pool1 * req_to_token_stride
            + st1 * r2t_stride1
        )
        if (ln0 <= BLOCK) and (ln1 <= BLOCK):
            # Fast path (typical): both windows fit one masked tile each;
            # both tile loads issue before both stores.
            m0 = cols < ln0
            m1 = cols < ln1
            v0 = tl.load(src0 + cols * r2t_stride1, mask=m0)
            v1 = tl.load(src1 + cols * r2t_stride1, mask=m1)
            tl.store(chunk_kv_indices_ptr + ob0 + cols, v0, mask=m0)
            tl.store(chunk_kv_indices_ptr + ob1 + cols, v1, mask=m1)
        else:
            full0 = ln0 // BLOCK
            for i in range(full0):
                off = i * BLOCK + cols
                data = tl.load(src0 + off * r2t_stride1)
                tl.store(chunk_kv_indices_ptr + ob0 + off, data)
            rest0 = ln0 - full0 * BLOCK
            if rest0 > 0:
                off = full0 * BLOCK + cols
                m = off < ln0
                data = tl.load(src0 + off * r2t_stride1, mask=m)
                tl.store(chunk_kv_indices_ptr + ob0 + off, data, mask=m)
            full1 = ln1 // BLOCK
            for i in range(full1):
                off = i * BLOCK + cols
                data = tl.load(src1 + off * r2t_stride1)
                tl.store(chunk_kv_indices_ptr + ob1 + off, data)
            rest1 = ln1 - full1 * BLOCK
            if rest1 > 0:
                off = full1 * BLOCK + cols
                m = off < ln1
                data = tl.load(src1 + off * r2t_stride1, mask=m)
                tl.store(chunk_kv_indices_ptr + ob1 + off, data, mask=m)


# Single-wave CTA dispatch ceiling on this board: grids above this pay
# ~48ns/CTA of dispatch wave quantization (measured 2026-11-08, see module
# docstring).  48 flat CTAs measured 4.10us; 64 flat CTAs 6.37us.
_FLAT_MAX_CTAS = 48


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
    for name in (
        "req_to_token",
        "req_pool_indices",
        "chunk_start_idx",
        "chunk_seq_lens",
        "chunk_cu_seq_lens",
        "chunk_kv_indices",
    ):
        v = kwargs.get(name)
        if isinstance(v, torch.Tensor):
            named[name] = v
    pos = [t for t in args if isinstance(t, torch.Tensor)]
    order = (
        "req_to_token",
        "req_pool_indices",
        "chunk_start_idx",
        "chunk_seq_lens",
        "chunk_cu_seq_lens",
        "chunk_kv_indices",
    )
    tensors = [named.get(n) for n in order]
    if len(pos) == 6 and any(t is None for t in tensors):
        tensors = pos
    if any(t is None for t in tensors):
        raise RuntimeError(
            f"create_chunked_prefix_cache_kv_indices: could not bind all 6 "
            f"tensors (named={sorted(named.keys())}, n_positional={len(pos)})"
        )
    (
        req_to_token,
        req_pool_indices,
        chunk_start_idx,
        chunk_seq_lens,
        chunk_cu_seq_lens,
        chunk_kv_indices,
    ) = tensors

    bs = req_pool_indices.shape[0]
    if bs == 0:
        return chunk_kv_indices

    # Device-time rule (q2, 2026-09-30): the wrapper may not create work.  A
    # `to(torch.int32)` and a `.contiguous()` on a sliced table are both real
    # device kernels and the platform bills device time only, so the four index
    # tensors go in at their own dtype (the kernel narrows with tl.to(int32),
    # which is free) and req_to_token goes in with its own strides.
    if req_to_token.dim() == 2:
        r2t = req_to_token
    else:
        r2t = req_to_token.contiguous().view(-1, req_to_token.shape[-1])
    s0 = r2t.stride(0)
    s1 = r2t.stride(1)

    def _keep(t):
        return t if t.is_contiguous() else t.contiguous()

    pool_c = _keep(req_pool_indices)
    start_c = _keep(chunk_start_idx)
    lens_c = _keep(chunk_seq_lens)
    cu_c = _keep(chunk_cu_seq_lens)
    out_c = (
        chunk_kv_indices
        if chunk_kv_indices.is_contiguous()
        else chunk_kv_indices.contiguous()
    )

    errors = []
    # enflame launch dispatch (2026-11-08 probe, do_bench KERNEL median):
    # - flat one-CTA-per-request grid while it fits a single dispatch wave;
    #   bs == 1 keeps the historical (2048, 4) launch, bs >= 2 uses nw=2
    #   (-0.60us at bs=8, -0.28us at bs=32, both > 2x noise band);
    # - beyond the wave ceiling a GROUP=2 persistent grid-stride kernel
    #   keeps the grid <= 48 CTAs (bs=64/128/256: -26%/-46%/-59%).
    if bs > _FLAT_MAX_CTAS:
        grid = (min(_FLAT_MAX_CTAS, (bs + 1) // 2),)
        for blk, nw in ((2048, 2), (1024, 2), (2048, 4), (1024, 4)):
            try:
                _ccpk_group2[grid](
                    r2t,
                    pool_c,
                    start_c,
                    lens_c,
                    cu_c,
                    out_c,
                    bs,
                    BLOCK=blk,
                    num_warps=nw,
                    req_to_token_stride=s0,
                    r2t_stride1=s1,
                )
                return out_c
            except Exception as e:  # noqa: BLE001
                errors.append(f"g{blk}_w{nw}: {type(e).__name__}: {e}")

    if bs == 1:
        orders = ((2048, 4), (2048, 2), (1024, 4), (4096, 4), (512, 4))
    else:
        orders = ((2048, 2), (2048, 4), (1024, 4), (4096, 4), (512, 4))
    for blk, nw in orders:
        try:
            _ccpk_rows[(bs,)](
                r2t,
                pool_c,
                start_c,
                lens_c,
                cu_c,
                out_c,
                BLOCK=blk,
                num_warps=nw,
                req_to_token_stride=s0,
                r2t_stride1=s1,
            )
            return out_c
        except Exception as e:  # noqa: BLE001
            errors.append(f"b{blk}_w{nw}: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"create_chunked_prefix_cache_kv_indices all variants failed "
        f"(device={req_to_token.device.type}, bs={bs}) | "
        + " || ".join(errors)
    )


__all__ = ["create_chunked_prefix_cache_kv_indices"]
