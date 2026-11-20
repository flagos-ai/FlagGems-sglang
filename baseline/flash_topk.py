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


"""Native SGLang DeepSeek V3.2 indexer benchmark without a loaded model."""

from types import SimpleNamespace

import torch
import triton

OP_NAME = "flash_topk"
DTYPES = [torch.float8_e4m3fn]
IS_INPLACE = False


def native():
    """Resolve the FP32 DeepGEMM and SGLang DSA v2 kernels."""
    try:
        import deep_gemm
        from sglang.kernels.ops.attention.dsv4.topk import (
            plan_topk_v2,
            topk_transform_paged_v2,
            topk_transform_ragged_v2,
        )

        try:
            from sglang.kernels.ops.kvcache.dsa_index_k_cache import GetKAndS
        except ImportError:
            from sglang.kernels.ops.attention.dsa.index_buf_accessor import (
                GetKAndS,
            )

        paged = deep_gemm.fp8_paged_mqa_logits
        ragged = deep_gemm.fp8_mqa_logits
        get_paged_plan = deep_gemm.get_paged_mqa_logits_metadata
    except (ImportError, AttributeError, OSError):
        return None

    def prepare(lengths, prefill, request_lengths, starts):
        if prefill:
            return request_lengths, starts, starts + lengths
        sm_count = torch.cuda.get_device_properties(
            lengths.device
        ).multi_processor_count
        lengths_2d = lengths.unsqueeze(1)
        score_plan = get_paged_plan(lengths_2d, 64, sm_count)
        select_plan = plan_topk_v2(lengths)
        return lengths_2d, score_plan, select_plan

    def run(
        query,
        weights,
        cache,
        table,
        requests,
        lengths,
        context,
        out,
        workspace,
        *,
        prefill,
        native_metadata,
        output_page_table=None,
        output_offsets=None,
    ):
        if prefill:
            request_lengths, starts, ends = native_metadata
            pool = SimpleNamespace(
                page_size=64,
                index_page_size=64,
                index_key_bytes_per_token=128,
                index_head_dim=128,
                quant_block_size=128,
            )
            gathered_k, gathered_scale = GetKAndS.execute(
                pool,
                cache,
                page_indices=table,
                seq_len_tensor=request_lengths,
                seq_len_sum=table.shape[0] * context,
                max_seq_len=context,
            )
            keys = gathered_k.view(torch.float8_e4m3fn)
            scales = gathered_scale.view(torch.float32).reshape(-1)
            logits = ragged(
                query,
                (keys, scales),
                weights,
                starts,
                ends,
                clean_logits=False,
            )
            topk_transform_ragged_v2(
                logits,
                lengths,
                out_offsets=output_offsets,
                out_indices=out,
                row_starts=starts,
            )
        else:
            lengths_2d, score_plan, select_plan = native_metadata
            logits = paged(
                query.unsqueeze(1),
                cache.view(-1, 64, 1, 132),
                weights,
                lengths_2d,
                table,
                score_plan,
                context,
                clean_logits=False,
            )
            topk_transform_paged_v2(
                logits,
                lengths,
                output_page_table,
                out,
                64,
                select_plan,
            )
        return out

    return SimpleNamespace(run=run, prepare=prepare)


def build_inputs(shape, dtype, device, workspace_size, prepare):
    requests, queries_per_request, context = shape
    rows = requests * queries_per_request
    prefill = queries_per_request > 1
    dtype_i = torch.int32
    rng = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(rows, 64, 128, device=device, generator=rng).to(dtype)
    weights = (
        torch.randn(rows, 64, device=device, generator=rng) / (64 * 128) ** 0.5
    )
    pages = triton.cdiv(context, 64)
    k = torch.randn(requests * pages, 8192, device=device, generator=rng)
    cache = torch.empty(
        (requests * pages, 8448), device=device, dtype=torch.uint8
    )
    cache[:, :8192].copy_(k.to(dtype).view(torch.uint8))
    cache[:, 8192:].view(torch.float32).uniform_(0.2, 1, generator=rng)
    table = torch.randperm(requests * pages, device=device, generator=rng)
    table = table.to(dtype_i).view(requests, pages)
    r = torch.arange(requests, device=device, dtype=dtype_i)
    row_map = r.repeat_interleave(queries_per_request)
    ext = torch.arange(queries_per_request, device=device, dtype=dtype_i)
    lengths = (context - queries_per_request + 1 + ext).repeat(requests)
    if not prefill:
        lengths -= r % 31
    out = torch.empty(rows, 2048, dtype=dtype_i, device=device)
    scratch = torch.empty(
        workspace_size(rows, context), dtype=dtype_i, device=device
    )
    starts = row_map * context
    req_lens = torch.full((requests,), context, device=device, dtype=dtype_i)
    metadata = prepare(lengths, prefill, req_lens, starts)
    kwargs = dict(
        prefill=prefill,
        native_metadata=metadata,
        output_offsets=starts if prefill else None,
        output_page_table=None if prefill else table,
    )
    return (
        q,
        weights,
        cache,
        table,
        row_map,
        lengths,
        context,
        out,
        scratch,
        kwargs,
    )
