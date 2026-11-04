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

"""context_attention baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.prefill_attention.context_attention_fwd
    调用形态复刻 problems/attention/context_attention/baseline.py：
        o = zeros_like(q)
        context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len,
                              max_input_len, is_causal=is_causal)
    结果写入预分配的 o（输出缓冲），故 IS_INPLACE=True。

输入构造复刻 problems/attention/context_attention/cases.py 的 BENCH_CASES：
    _case([seq_len]*bs, 32, 128, True) for bs in (1,8,64) for seq_len in (128,2048)
    q/k/v = randn(total, num_heads, head_dim)（dtype 直接 float32，见 cases）；
    b_start_loc / b_seq_len 由 seq_lens 构造；is_causal=True。
dtype 取 benchmark 的 [float32]。
"""

import torch

OP_NAME = "context_attention"
DTYPES = [torch.float32]
IS_INPLACE = True  # 结果写入预分配的 o 缓冲

_NUM_HEADS = 32
_HEAD_DIM = 128
# BENCH_CASES: bs in (1, 8, 64), seq_len in (128, 2048)
_CASES = [(bs, sl) for bs in (1, 8, 64) for sl in (128, 2048)]


def native():
    """解析 sglang.kernels.ops.attention.prefill_attention.context_attention_fwd。"""
    try:
        from sglang.kernels.ops.attention.prefill_attention import (
            context_attention_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "seq_len": sl, "num_heads": _NUM_HEADS, "head_dim": _HEAD_DIM}
        for (bs, sl) in _CASES
    ]


def build_inputs(binding, dtype, device):
    bs, sl = binding["bs"], binding["seq_len"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    seq_lens = [sl] * bs
    total = sum(seq_lens)
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(total, num_heads, head_dim, generator=g, device=device, dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, generator=g, device=device, dtype=dtype)
    v = torch.randn(total, num_heads, head_dim, generator=g, device=device, dtype=dtype)

    b_start_loc = torch.zeros(len(seq_lens), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens[:-1], dtype=torch.int32, device=device), 0
    )
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_input_len = max(seq_lens)

    o = torch.zeros_like(q)
    return (q, k, v, o, b_start_loc, b_seq_len, max_input_len), {"is_causal": True}


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
            "o": {"shape": [total, num_heads, head_dim], "dtype": dt,
                  "note": "原地写回"},
            "b_start_loc": {"shape": [bs], "dtype": "torch.int32"},
            "b_seq_len": {"shape": [bs], "dtype": "torch.int32"},
            "max_input_len": {"scalar": sl},
            "is_causal": {"scalar": True},
        },
        "outputs": {
            "o": {"shape": [total, num_heads, head_dim], "dtype": dt},
        },
        "dims": {
            "bs": bs, "seq_len": sl, "total_tokens": total,
            "num_heads": num_heads, "head_dim": head_dim,
        },
    }
