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

_MAX_BLOCK_S = 512
_MAX_NCHUNK = 4
_MAX_CHUNK_SPAN = _MAX_BLOCK_S * _MAX_NCHUNK


@triton.autotune(
    configs=tuple(
        triton.Config(
            {"BLOCK_S": bs, "NCHUNK": nch}, num_warps=w, num_stages=s
        )
        for bs, nch, w, s in (
            (512, 2, 8, 1),
            (512, 2, 8, 2),
            (512, 1, 8, 2),
            (512, 4, 8, 1),
            (1024, 2, 8, 1),
            (512, 2, 4, 1),
        )
    ),
    key=["S", "RC"],
)
@triton.jit
def _state_scatter_with_mask_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_ptr,
    S,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_cache_stride,
    RC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NCHUNK: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_rl = tl.program_id(1)
    req = pid_rl % RC
    layer = pid_rl // RC

    step = tl.load(step_ptr + req)
    if step < 0:
        return
    dst_loc = tl.load(dst_idx_ptr + req)

    src_base = (
        src_ptr
        + layer.to(tl.int64) * src_layer_stride
        + req.to(tl.int64) * src_req_stride
        + step.to(tl.int64) * src_step_stride
    )
    dst_base = (
        dst_ptr
        + layer.to(tl.int64) * dst_layer_stride
        + dst_loc.to(tl.int64) * dst_cache_stride
    )

    offs = pid_s * (BLOCK_S * NCHUNK) + tl.arange(0, BLOCK_S)
    if EVEN_S:
        for _ in tl.static_range(NCHUNK):
            vals = tl.load(src_base + offs)
            tl.store(dst_base + offs, vals)
            offs += BLOCK_S
    else:
        for _ in tl.static_range(NCHUNK):
            mask = offs < S
            vals = tl.load(src_base + offs, mask=mask)
            tl.store(dst_base + offs, vals, mask=mask)
            offs += BLOCK_S


def state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    if not (dst.is_contiguous() and src.is_contiguous()):
        src_c = src.contiguous()
        dst_c = dst.contiguous()
        _launch(dst_c, src_c, dst_indices_raw, step_indices_raw)
        if dst_c.data_ptr() != dst.data_ptr():
            dst.copy_(dst_c)
        return dst
    return _launch(dst, src, dst_indices_raw, step_indices_raw)


def _launch(dst, src, dst_indices_raw, step_indices_raw):
    layers = dst.shape[0]
    sequences = src.shape[1]
    state_elems = dst.numel() // (layers * dst.shape[1])
    even_s = state_elems % _MAX_CHUNK_SPAN == 0

    def grid(meta):
        return (
            triton.cdiv(state_elems, meta["BLOCK_S"] * meta["NCHUNK"]),
            layers * sequences,
        )

    _state_scatter_with_mask_kernel[grid](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        state_elems,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        RC=sequences,
        EVEN_S=even_s,
    )
    return dst


__all__ = ["state_scatter_with_mask"]
