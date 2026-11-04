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

"""pad_draft_extend_query baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.pad.pad_draft_extend_query
    调用形态复刻 problems/attention/pad_draft_extend_query/baseline.py：
        out = padded_q.clone()
        pad_draft_extend_query(q, out, seq_lens_q, cu_seqlens_q)
    kernel 原地写入传入的 out（故 baseline 先 clone padded_q）。IS_INPLACE=True
    （第二个实参 out 被原地写回）。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    seq_lens = randint(1, max_seq_len+1, (bs,), int32)，cu 前缀和，total=cu[-1]
    q = randn(total, num_heads, head_dim).bf16
    padded_q = zeros(bs, max_seq_len, num_heads, head_dim).bf16
    max_seq_len=8, num_heads=16, head_dim=128。seed=7。total 数据相关。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "pad_draft_extend_query"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 out（padded_q 的克隆）

_MAX_SEQ_LEN = 8
_NUM_HEADS = 16
_HEAD_DIM = 128
_CASES = [1, 8, 64, 256, 1024]


def native():
    """解析 sglang.kernels.ops.attention.pad.pad_draft_extend_query。"""
    try:
        from sglang.kernels.ops.attention.pad import pad_draft_extend_query as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "max_seq_len": _MAX_SEQ_LEN,
         "num_heads": _NUM_HEADS, "head_dim": _HEAD_DIM}
        for bs in _CASES
    ]


def build_inputs(binding, dtype, device):
    bs = binding["bs"]
    max_seq_len, num_heads, head_dim = (
        binding["max_seq_len"], binding["num_heads"], binding["head_dim"],
    )
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1, max_seq_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    total = int(cu[-1])
    q = torch.randn(total, num_heads, head_dim, device=device, generator=g).to(dtype)
    padded_q = torch.zeros(
        bs, max_seq_len, num_heads, head_dim, dtype=dtype, device=device
    )
    out = padded_q.clone()
    return (q, out, seq_lens, cu), {}


def key_shape(binding):
    return "bs%d_msl%d_nh%d_hd%d" % (
        binding["bs"], binding["max_seq_len"],
        binding["num_heads"], binding["head_dim"],
    )


def config(binding, dtype):
    bs = binding["bs"]
    max_seq_len, num_heads, head_dim = (
        binding["max_seq_len"], binding["num_heads"], binding["head_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": ["total_q", num_heads, head_dim], "dtype": dt},
            "out": {"shape": [bs, max_seq_len, num_heads, head_dim], "dtype": dt,
                    "note": "原地写回"},
            "seq_lens_q": {"shape": [bs], "dtype": "torch.int32"},
            "cu_seqlens_q": {"shape": [bs + 1], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [bs, max_seq_len, num_heads, head_dim], "dtype": dt},
        },
        "dims": {
            "bs": bs, "max_seq_len": max_seq_len,
            "num_heads": num_heads, "head_dim": head_dim,
            "note": "total_q = sum(randint(1,max_seq_len+1)) 数据相关（seed=7）",
        },
    }
