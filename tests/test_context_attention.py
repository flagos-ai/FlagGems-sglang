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
import torch.nn.functional as F

import flaggems_sglang
import flaggems_sglang.ops.context_attention as context_attention_module

from . import conftest as cfg

CASES = [
    ([1], 2, 2, 32),
    ([3, 17], 4, 4, 64),
    ([65, 7, 33], 4, 4, 96),
    ([1, 63], 8, 1, 128),
]


def test_context_attention_launch_grid_is_bounded():
    # Reproduce the official Ascend failure product: 2,112 * 32 = 67,584.
    q_programs, batch_heads, batch_heads_per_launch = (
        context_attention_module._launch_plan(
            total_tokens=135168,
            batch_size=1,
            q_heads=32,
            block_m=64,
            max_input_len=135168,
        )
    )
    assert q_programs == 2112
    assert batch_heads == 32
    assert (
        q_programs * batch_heads_per_launch
        <= context_attention_module._MAX_GRID_PROGRAMS
    )
    assert batch_heads_per_launch == 31


def _reference(q, k, v, starts, lengths, is_causal):
    out = torch.empty_like(q, dtype=torch.float32)
    group_size = q.shape[1] // k.shape[1]
    for start_tensor, length_tensor in zip(starts, lengths):
        start = int(start_tensor.item())
        length = int(length_tensor.item())
        end = start + length
        q_seq = q[start:end].permute(1, 0, 2).float()
        k_seq = k[start:end].permute(1, 0, 2).float()
        v_seq = v[start:end].permute(1, 0, 2).float()
        if group_size != 1:
            k_seq = k_seq.repeat_interleave(group_size, dim=0)
            v_seq = v_seq.repeat_interleave(group_size, dim=0)
        out[start:end] = F.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, is_causal=is_causal
        ).permute(1, 0, 2)
    return out


@pytest.mark.parametrize("seq_lens,q_heads,kv_heads,head_dim", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("max_input_len_mode", ["exact", "underreported"])
@pytest.mark.context_attention
def test_context_attention(
    seq_lens,
    q_heads,
    kv_heads,
    head_dim,
    dtype,
    is_causal,
    max_input_len_mode,
):
    device = cfg.device
    total_tokens = sum(seq_lens)
    starts = torch.tensor(
        [sum(seq_lens[:i]) for i in range(len(seq_lens))],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    torch.manual_seed(20260823 + total_tokens + head_dim)
    q = torch.randn(
        total_tokens, q_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        total_tokens, kv_heads, head_dim, dtype=dtype, device=device
    )
    v = torch.randn_like(k)

    expected = _reference(q, k, v, starts, lengths, is_causal)
    max_input_len = max(seq_lens) if max_input_len_mode == "exact" else 1
    actual = flaggems_sglang.context_attention(
        q, k, v, starts, lengths, max_input_len, is_causal
    )

    assert actual.shape == q.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
