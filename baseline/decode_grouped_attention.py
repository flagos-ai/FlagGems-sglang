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

"""decode_grouped_attention baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.decode_attention.decode_attention_fwd_grouped
    调用形态复刻 problems/attention/decode_grouped_attention/baseline.py：
        num_kv_splits = full((B,), 4, int32)
        o = zeros(B, H_Q, D_V)
        attn_logits = empty(B, H_Q, MAX_KV_SPLITS, D_V, fp32)
        attn_lse = empty(B, H_Q, MAX_KV_SPLITS, fp32)
        decode_attention_fwd_grouped(
            q, k_buffer, v_buffer, o, kv_indptr, kv_indices,
            attn_logits, attn_lse, num_kv_splits, MAX_KV_SPLITS, sm_scale, 1.0)
    结果写入预分配的 o / attn_logits / attn_lse 缓冲，故 IS_INPLACE=True。

输入构造复刻 problems/attention/decode_grouped_attention/cases.py 的 BENCH_CASES（lazy）：
    _case(B, 128, 1, 128, seq_len) for (B, seq_len) in
    ((1,2048),(8,2048),(64,512),(512,128),(4096,128))
    —— 高 H_Q:H_KV（128:1）走 grouped kernel。
dtype 取 benchmark 的 [bfloat16]。
"""

import torch

OP_NAME = "decode_grouped_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 结果写入预分配的 o / attn_logits / attn_lse 缓冲

_MAX_KV_SPLITS = 8
_H_Q = 128
_H_KV = 1
_D = 128
# BENCH_CASES: (B, seq_len)
_CASES = [(1, 2048), (8, 2048), (64, 512), (512, 128), (4096, 128)]


def native():
    """解析 sglang.kernels.ops.attention.decode_attention.decode_attention_fwd_grouped。"""
    try:
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd_grouped as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"B": b, "seq_len": sl, "H_Q": _H_Q, "H_KV": _H_KV, "D": _D}
        for (b, sl) in _CASES
    ]


def build_inputs(binding, dtype, device):
    B, seq_len = binding["B"], binding["seq_len"]
    H_Q, H_KV, D = binding["H_Q"], binding["H_KV"], binding["D"]
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D ** 0.5)
    g = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(B, H_Q, D, generator=g, device=device, dtype=torch.float32).to(dtype)
    k_buffer = torch.randn(
        total_tokens, H_KV, D, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    v_buffer = torch.randn(
        total_tokens, H_KV, D, generator=g, device=device, dtype=torch.float32
    ).to(dtype)

    b_seq_len = torch.full((B,), seq_len, device=device)
    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)

    D_V = v_buffer.shape[-1]
    num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device=device)
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
    attn_logits = torch.empty(
        (B, H_Q, _MAX_KV_SPLITS, D_V), dtype=torch.float32, device=device
    )
    attn_lse = torch.empty((B, H_Q, _MAX_KV_SPLITS), dtype=torch.float32, device=device)

    return (
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        _MAX_KV_SPLITS,
        sm_scale,
        1.0,
    ), {}


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
            "attn_logits": {"shape": [B, H_Q, _MAX_KV_SPLITS, D],
                            "dtype": "torch.float32", "note": "原地写回（scratch）"},
            "attn_lse": {"shape": [B, H_Q, _MAX_KV_SPLITS],
                         "dtype": "torch.float32", "note": "原地写回（scratch）"},
            "num_kv_splits": {"shape": [B], "dtype": "torch.int32"},
            "max_kv_splits": {"scalar": _MAX_KV_SPLITS},
            "sm_scale": {"scalar": 1.0 / (D ** 0.5)},
            "logit_cap": {"scalar": 1.0},
        },
        "outputs": {
            "o": {"shape": [B, H_Q, D], "dtype": dt},
        },
        "dims": {
            "B": B, "seq_len": seq_len, "total_tokens": total_tokens,
            "H_Q": H_Q, "H_KV": H_KV, "D": D, "max_kv_splits": _MAX_KV_SPLITS,
        },
    }
