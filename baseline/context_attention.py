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

"""Native SGLang collection workload for context_attention."""

import torch

OP_NAME = "context_attention"
DTYPES = [torch.float32]
IS_INPLACE = True

MUTATED_INPUTS = (3,)

IS_CAUSAL = True
SHAPES = [
    (bs, seq_len, 32, 128) for bs in (1, 8, 64) for seq_len in (128, 2048)
]
MORE_SHAPES = [(2, 8, 4, 128), (3, 30, 4, 96), (1, 20, 4, 80), (1, 9, 4, 13)]


def native():
    """解析 sglang.kernels.ops.attention.prefill_attention.context_attention_fwd。"""
    try:
        from sglang.kernels.ops.attention.prefill_attention import (
            context_attention_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": bs,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        for (bs, seq_len, num_heads, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, seq_len, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = [seq_len] * bs
    total = sum(seq_lens)
    q, k, v = (
        torch.randn(
            total,
            num_heads,
            head_dim,
            generator=g,
            device=device,
            dtype=cur_dtype,
        )
        for _ in range(3)
    )
    b_start_loc = torch.zeros(bs, dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens[:-1], dtype=torch.int32, device=device), 0
    )
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return q, k, v, b_start_loc, b_seq_len, max(seq_lens), IS_CAUSAL


def build_inputs(binding, dtype, device):
    shape = (
        binding["bs"],
        binding["seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    )
    q, k, v, starts, lengths, max_len, causal = build_case(
        shape, dtype, device
    )
    return (q, k, v, torch.zeros_like(q), starts, lengths, max_len), {
        "is_causal": causal
    }


def key_shape(binding):
    return "bs%d_sl%d_h%d_d%d" % (
        binding["bs"],
        binding["seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    )


def config(binding, dtype):
    bs, sl = binding["bs"], binding["seq_len"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    total = sl * bs
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [total, num_heads, head_dim], "dtype": dt},
            "k": {"shape": [total, num_heads, head_dim], "dtype": dt},
            "v": {"shape": [total, num_heads, head_dim], "dtype": dt},
            "o": {
                "shape": [total, num_heads, head_dim],
                "dtype": dt,
                "note": "原地写回",
            },
            "b_start_loc": {"shape": [bs], "dtype": "torch.int32"},
            "b_seq_len": {"shape": [bs], "dtype": "torch.int32"},
            "max_input_len": {"scalar": sl},
            "is_causal": {"scalar": True},
        },
        "outputs": {
            "o": {"shape": [total, num_heads, head_dim], "dtype": dt},
        },
        "dims": {
            "bs": bs,
            "seq_len": sl,
            "total_tokens": total,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
    }
