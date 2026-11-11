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

"""Native SGLang collection workload for decode_grouped_attention."""

import torch

if __package__:
    from .__decode_inputs import decode_inputs as _decode_inputs
else:
    from __decode_inputs import decode_inputs as _decode_inputs

OP_NAME = "decode_grouped_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (3, 6, 7)

_MAX_KV_SPLITS = 8

SHAPES = [
    (B, 128, 1, 128, seq_len)
    for B, seq_len in (
        (1, 2048),
        (8, 2048),
        (64, 512),
        (512, 128),
        (4096, 128),
    )
]
MORE_SHAPES = [
    (2, 32, 1, 64, 10),
    (2, 128, 1, 64, 10),
    (2, 32, 4, 80, 10),
    (2, 128, 8, 512, 128),
]


def native():
    """解析 sglang.kernels.ops.attention.decode_attention.decode_attention_fwd_grouped。"""
    try:
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd_grouped as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"B": B, "H_Q": H_Q, "H_KV": H_KV, "D": D, "seq_len": seq_len}
        for (B, H_Q, H_KV, D, seq_len) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    B, H_Q, H_KV, D, seq_len = shape
    g = torch.Generator(device=device).manual_seed(0)
    total_tokens = B * seq_len
    q = torch.randn(
        B, H_Q, D, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    k_buffer = torch.randn(
        total_tokens, H_KV, D, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    v_buffer = torch.randn(
        total_tokens, H_KV, D, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # Every request holds seq_len contiguous KV slots.
    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.full((B,), seq_len, dtype=torch.int32, device=device), dim=0
    )
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
    return q, k_buffer, v_buffer, kv_indptr, kv_indices, 1.0 / (D**0.5)


def build_inputs(binding, dtype, device):
    shape = (
        binding["B"],
        binding["H_Q"],
        binding["H_KV"],
        binding["D"],
        binding["seq_len"],
    )
    return _decode_inputs(build_case(shape, dtype, device), device)


def key_shape(binding):
    return "B%d_sl%d_hq%d_hkv%d_d%d" % (
        binding["B"],
        binding["seq_len"],
        binding["H_Q"],
        binding["H_KV"],
        binding["D"],
    )


def config(binding, dtype):
    B, seq_len = binding["B"], binding["seq_len"]
    H_Q, H_KV, D = binding["H_Q"], binding["H_KV"], binding["D"]
    total_tokens = B * seq_len
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [B, H_Q, D], "dtype": dt},
            "k_buffer": {"shape": [total_tokens, H_KV, D], "dtype": dt},
            "v_buffer": {"shape": [total_tokens, H_KV, D], "dtype": dt},
            "o": {"shape": [B, H_Q, D], "dtype": dt, "note": "原地写回"},
            "kv_indptr": {"shape": [B + 1], "dtype": "torch.int32"},
            "kv_indices": {"shape": [total_tokens], "dtype": "torch.int32"},
            "attn_logits": {
                "shape": [B, H_Q, _MAX_KV_SPLITS, D],
                "dtype": "torch.float32",
                "note": "原地写回（scratch）",
            },
            "attn_lse": {
                "shape": [B, H_Q, _MAX_KV_SPLITS],
                "dtype": "torch.float32",
                "note": "原地写回（scratch）",
            },
            "num_kv_splits": {"shape": [B], "dtype": "torch.int32"},
            "max_kv_splits": {"scalar": _MAX_KV_SPLITS},
            "sm_scale": {"scalar": 1.0 / (D**0.5)},
            "logit_cap": {"scalar": 0.0},
        },
        "outputs": {
            "o": {"shape": [B, H_Q, D], "dtype": dt},
        },
        "dims": {
            "B": B,
            "seq_len": seq_len,
            "total_tokens": total_tokens,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D": D,
            "max_kv_splits": _MAX_KV_SPLITS,
        },
    }
