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


from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

if not all(
    hasattr(tle.gpu, name) for name in ("alloc", "wgmma", "wgmma_wait")
):
    raise ImportError("FlashTopK requires shared allocation and WGMMA in TLE")


@triton.jit
def _score_request(
    query,
    cache,
    k_scales,
    weights,
    table,
    scores,
    first,
    request,
    lens,
    group,
    table_rows: tl.constexpr,
    cache_pages: tl.constexpr,
    table_cols: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    rows: tl.constexpr,
    context: tl.constexpr,
    pitch: tl.constexpr,
    query_stride: tl.constexpr,
    weights_stride: tl.constexpr,
    pages_per_cta: tl.constexpr,
    row_group: tl.constexpr,
):
    start = group * pages_per_cta
    longest = tl.max(lens, 0)
    if (
        (start * 64 < longest)
        & (longest > 2048)
        & (request >= 0)
        & (request < table_rows)
    ):
        heads = tl.arange(0, row_group * 64)
        dims = tl.arange(0, 128)
        rows_h = first + heads // 64
        q = tl.load(
            query
            + rows_h[:, None].to(tl.int64) * query_stride
            + (heads[:, None] % 64) * 128
            + dims[None, :],
            rows_h[:, None] < rows,
            0,
        ).to(tl.float8e4nv, bitcast=True)
        q_smem = tle.gpu.alloc(
            [row_group * 64, 128],
            dtype=tl.float8e4nv,
            init_value=q,
            scope=tle.gpu.smem,
        )
        gate = tl.load(
            weights + rows_h.to(tl.int64) * weights_stride + heads % 64,
            rows_h < rows,
            0.0,
        )
        tok = tl.arange(0, 64)
        local_rows = first + tl.arange(0, row_group)
        # TMA needs a real full page. On the tail, fetch a safe page and mask
        # its output rather than allowing an out-of-range table access.
        for step in tl.range(0, pages_per_cta, num_stages=2):
            logical_page = start + step
            valid = (logical_page < table_cols) & (logical_page * 64 < longest)
            page = tl.load(
                table
                + request.to(tl.int64) * table_row_stride
                + logical_page * table_col_stride,
                valid,
                0,
            )
            valid_page = valid & (page >= 0) & (page < cache_pages)
            page = tl.where(valid_page, page, 0)
            key = cache.load([page * 66, 0])
            pending = tle.gpu.wgmma(
                key, q_smem, trans_b=True, out_dtype=tl.float32
            )
            dots = tle.gpu.wgmma_wait(0, pending)
            weighted = tl.reshape(
                tl.maximum(dots, 0.0) * gate[None, :], (64, row_group, 64)
            )
            reduced = tl.trans(tl.sum(weighted, axis=2))
            scale = tl.load(k_scales + page.to(tl.int64) * 2112 + tok)
            col = logical_page * 64 + tok
            tl.store(
                scores
                + local_rows[:, None].to(tl.int64) * pitch
                + col[None, :],
                tl.where(valid_page, reduced * scale[None, :], -float("inf")),
                (local_rows[:, None] < rows)
                & (col[None, :] < lens[:, None])
                & (col[None, :] < context),
            )


@triton.jit
def _score_paged(
    query,
    cache,
    k_scales,
    weights,
    table,
    row_to_request,
    lengths,
    scores,
    rows: tl.constexpr,
    context: tl.constexpr,
    pitch: tl.constexpr,
    query_stride: tl.constexpr,
    weights_stride: tl.constexpr,
    table_rows: tl.constexpr,
    cache_pages: tl.constexpr,
    table_cols: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    request_stride: tl.constexpr,
    length_stride: tl.constexpr,
    pages_per_cta: tl.constexpr,
    row_group: tl.constexpr,
):
    first = tl.program_id(1) * row_group
    group = tl.program_id(0)
    row = first + tl.arange(0, row_group)
    lens = tl.load(lengths + row * length_stride, row < rows, 0)
    lens = tl.where((lens >= 0) & (lens <= context), lens, 0)
    request = tl.load(row_to_request + first * request_stride)
    if row_group == 2:
        request_1 = tl.load(
            row_to_request + (first + 1) * request_stride,
            first + 1 < rows,
            request,
        )
        if (request != request_1) & (first + 1 < rows):
            # A dynamic row pair may straddle two requests after graph replay.
            lane = tl.arange(0, 1)
            lens_0 = tl.gather(lens, lane, axis=0)
            lens_1 = tl.gather(lens, lane + 1, axis=0)
            _score_request(
                query,
                cache,
                k_scales,
                weights,
                table,
                scores,
                first,
                request,
                lens_0,
                group,
                table_rows,
                cache_pages,
                table_cols,
                table_row_stride,
                table_col_stride,
                rows,
                context,
                pitch,
                query_stride,
                weights_stride,
                pages_per_cta,
                1,
            )
            _score_request(
                query,
                cache,
                k_scales,
                weights,
                table,
                scores,
                first + 1,
                request_1,
                lens_1,
                group,
                table_rows,
                cache_pages,
                table_cols,
                table_row_stride,
                table_col_stride,
                rows,
                context,
                pitch,
                query_stride,
                weights_stride,
                pages_per_cta,
                1,
            )
        else:
            _score_request(
                query,
                cache,
                k_scales,
                weights,
                table,
                scores,
                first,
                request,
                lens,
                group,
                table_rows,
                cache_pages,
                table_cols,
                table_row_stride,
                table_col_stride,
                rows,
                context,
                pitch,
                query_stride,
                weights_stride,
                pages_per_cta,
                row_group,
            )
    else:
        _score_request(
            query,
            cache,
            k_scales,
            weights,
            table,
            scores,
            first,
            request,
            lens,
            group,
            table_rows,
            cache_pages,
            table_cols,
            table_row_stride,
            table_col_stride,
            rows,
            context,
            pitch,
            query_stride,
            weights_stride,
            pages_per_cta,
            row_group,
        )


def score_paged(
    query,
    cache,
    weights,
    table,
    row_to_request,
    lengths,
    scores,
    context,
    pitch,
    *,
    prefill,
):
    rows = query.shape[0]
    # Fixed launch parameters for the Hopper TMA/WGMMA path. Prefill's wider
    # queries share cache pages and keep the stationary Q tile in shared memory.
    query_group = 2 if prefill else 1
    pages = 16 if prefill and rows > 64 else 4
    if not prefill and rows <= 8 and context <= 8192:
        pages = 1
    cache_descriptor = TensorDescriptor.from_tensor(
        cache.view(query.dtype).view(-1, 128), [64, 128]
    )
    k_scales = cache.view(scores.dtype)[:, 2048:]
    _score_paged[
        (triton.cdiv(context, pages * 64), triton.cdiv(rows, query_group))
    ](
        query.view(torch.uint8),
        cache_descriptor,
        k_scales,
        weights,
        table,
        row_to_request,
        lengths,
        scores,
        rows,
        context,
        pitch,
        query.stride(0),
        weights.stride(0),
        table.shape[0],
        cache.shape[0],
        table.shape[1],
        *table.stride(),
        row_to_request.stride(0),
        lengths.stride(0),
        pages,
        query_group,
        num_warps=4,
        enable_fp_fusion=False,
    )
