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

"""fused_softmax_pool baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.deepseek_v4_rope.fused_softmax_pool_triton
    调用形态复刻 problems/attention/fused_softmax_pool/baseline.py：
        fused_softmax_pool_triton(kv_score, head_dim)
    返回新张量，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/attention/fused_softmax_pool/cases.py 的 BENCH_CASES：
    kv_score = randn(bs, k, 2*head_dim).bf16，head_dim=128。seed=0。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_softmax_pool"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_HEAD_DIM = 128
# BENCH_CASES: (bs, k) for bs in (1,64,1024,4096) for k in (4,16,64)
_CASES = [(bs, k) for bs in (1, 64, 1024, 4096) for k in (4, 16, 64)]


def native():
    """解析 deepseek_v4_rope.fused_softmax_pool_triton。"""
    try:
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            fused_softmax_pool_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "k": k, "head_dim": _HEAD_DIM} for (bs, k) in _CASES]


def build_inputs(binding, dtype, device):
    bs, k, head_dim = binding["bs"], binding["k"], binding["head_dim"]
    g = torch.Generator(device=device).manual_seed(0)
    kv_score = torch.randn(
        bs, k, 2 * head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    return (kv_score, head_dim), {}


def key_shape(binding):
    return [binding["bs"], binding["k"], 2 * binding["head_dim"]]


def config(binding, dtype):
    bs, k, head_dim = binding["bs"], binding["k"], binding["head_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "kv_score": {"shape": [bs, k, 2 * head_dim], "dtype": dt},
            "head_dim": {"scalar": head_dim},
        },
        "outputs": {
            "out": {"shape": [bs, k, head_dim], "dtype": dt},
        },
        "dims": {"bs": bs, "k": k, "head_dim": head_dim},
    }
