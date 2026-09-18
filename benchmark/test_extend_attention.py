# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for attention/extend_attention."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/extend_attention.
SHAPES = [
    (B, n_ctx, 32, 8, 128)
    for B, n_ctx in ((1, 2048), (8, 2048), (64, 512), (256, 256))
]
MORE_SHAPES = [(4, 256, 12, 4, 128), (4, 256, 12, 4, 80), (2, 128, 8, 8, 64)]


def _input_fn(shape, cur_dtype, device):
    B, n_ctx, H_Q, H_KV, D = shape
    g = torch.Generator(device=device).manual_seed(0)

    def rnd_int(lo, hi, n):
        return torch.randint(
            lo, hi, (n,), dtype=torch.int32, device=device, generator=g
        )

    b_seq_len_prefix = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len_extend = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(b_seq_len_prefix, 0)
    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum()),), dtype=torch.int32, device=device
    )
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i].item(),
            (b_start_loc[i] + b_seq_len_prefix[i]).item(),
            device=device,
        )

    total_token_num = int(b_seq_len.sum())
    extend_token_num = int(b_seq_len_extend.sum())
    k_buffer = torch.randn(
        total_token_num,
        H_KV,
        D,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    v_buffer = torch.randn(
        total_token_num,
        H_KV,
        D,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)

    k_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=cur_dtype, device=device
    )
    v_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=cur_dtype, device=device
    )
    q_extend = torch.empty(
        (extend_token_num, H_Q, D), dtype=cur_dtype, device=device
    )
    for i in range(B):
        eib = (b_start_loc[i] + b_seq_len_prefix[i]).item()
        eie = (b_start_loc[i] + b_seq_len[i]).item()
        es = b_start_loc_extend[i].item()
        ee = (b_start_loc_extend[i] + b_seq_len_extend[i]).item()
        k_extend[es:ee] = k_buffer[eib:eie]
        v_extend[es:ee] = v_buffer[eib:eie]
        q_extend[es:ee] = torch.randn(
            (ee - es, H_Q, D),
            generator=g,
            device=device,
            dtype=torch.float32,
        ).to(cur_dtype)

    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, 0)
    max_len_extend = int(b_seq_len_extend.max())

    yield (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        max_len_extend,
    )


@pytest.mark.extend_attention
def test_perf_extend_attention():
    bench = OpBenchmark(
        op_name="extend_attention",
        torch_op=get_reference("extend_attention"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="B, n_ctx, H_Q, H_KV, D",
    )
    bench.set_gems(flaggems_sglang.extend_attention)
    bench.run()
