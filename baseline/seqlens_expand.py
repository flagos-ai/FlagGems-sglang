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

"""seqlens_expand baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.pad.seqlens_expand_triton
    调用形态复刻 problems/attention/seqlens_expand/baseline.py：
        seqlens_expand_triton(extend_seq_lens, seq_lens, total_len, max_q_len)
    返回新张量，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    qo = randint(1, max_q+1, (n,), int32)；kv = randint(1, 8192, (n,), int32)
    include_short_kv=True -> kv[::max(n//4,1)] = 0
    total_len = int(qo.sum())，max_q_len = int(qo.max())，max_q=16。seed=7。
dtype：输入为 int32 索引张量，DTYPES 取 [int32]。total_len/max_q_len 数据相关。
"""

import torch

OP_NAME = "seqlens_expand"
DTYPES = [torch.int32]
IS_INPLACE = False

_MAX_Q = 16
_CASES = [1, 8, 64, 512, 4096]


def native():
    """解析 sglang.kernels.ops.attention.pad.seqlens_expand_triton。"""
    try:
        from sglang.kernels.ops.attention.pad import seqlens_expand_triton as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"n": n, "max_q": _MAX_Q} for n in _CASES]


def build_inputs(binding, dtype, device):
    n, max_q = binding["n"], binding["max_q"]
    g = torch.Generator(device=device).manual_seed(7)
    qo = torch.randint(1, max_q + 1, (n,), dtype=torch.int32, device=device, generator=g)
    kv = torch.randint(1, 8192, (n,), dtype=torch.int32, device=device, generator=g)
    kv[:: max(n // 4, 1)] = 0
    total_len = int(qo.sum())
    max_q_len = int(qo.max())
    return (qo, kv, total_len, max_q_len), {}


def key_shape(binding):
    return [binding["n"]]


def config(binding, dtype):
    n, max_q = binding["n"], binding["max_q"]
    return {
        "inputs": {
            "extend_seq_lens": {"shape": [n], "dtype": "torch.int32"},
            "seq_lens": {"shape": [n], "dtype": "torch.int32"},
            "total_len": {"note": "标量 int = sum(extend_seq_lens)，数据相关"},
            "max_q_len": {"note": "标量 int = max(extend_seq_lens)，数据相关"},
        },
        "outputs": {
            "out": {"shape": ["total_len"], "dtype": "torch.int32",
                    "note": "按 qo 展开，长度 total_len 数据相关"},
        },
        "dims": {"n": n, "max_q": max_q,
                 "note": "total_len/max_q_len 数据相关（seed=7）"},
    }
