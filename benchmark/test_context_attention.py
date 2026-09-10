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

import pytest
import torch

import flaggems_sglang

from .attri_util import CONTEXT_ATTENTION_BENCH_SHAPES


@pytest.mark.parametrize(
    "seq_lens,num_heads,head_dim", CONTEXT_ATTENTION_BENCH_SHAPES
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.context_attention
def test_context_attention(
    seq_lens, num_heads, head_dim, dtype, is_causal, benchmark
):
    device = flaggems_sglang.device
    total_tokens = sum(seq_lens)
    starts = torch.tensor(
        [sum(seq_lens[:i]) for i in range(len(seq_lens))],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    q = torch.randn(
        total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    benchmark(
        flaggems_sglang.context_attention,
        q,
        k,
        v,
        starts,
        lengths,
        max(seq_lens),
        is_causal,
    )
