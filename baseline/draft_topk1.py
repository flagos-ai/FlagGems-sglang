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

"""draft_topk1 baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.speculative.topk1.draft_topk1_postprocess
    调用形态复刻 problems/speculative/draft_topk1/baseline.py：
        draft_topk1_postprocess(next_token_logits, positions,
            draft_tokens=draft_tokens, draft_token_column=draft_token_column)
    native 原地推进 positions 并（可选）写 draft_tokens（baseline 先 clone 两者）；
    故 IS_INPLACE=True，采样前由 build_inputs 重建实参。

输入构造复刻 problems/speculative/draft_topk1/cases.py 的 BENCH_CASES：
    _case(bs, 151936, with_draft_tokens=True)：
    next_token_logits=randn(bs, V) float32；positions=randint(0,4096,(bs,)) int64；
    draft_tokens=randint(0,V,(bs,4)) int64；draft_token_column=0。
dtype 取 benchmark/test_draft_topk1.py 的 [float32]。
"""

import torch

OP_NAME = "draft_topk1"
DTYPES = [torch.float32]
IS_INPLACE = True  # native 原地推进 positions / 写 draft_tokens

# BENCH_CASES: _case(bs, 151936, with_draft_tokens=True) for bs in (1,8,64,512)
_SHAPES = [
    {"bs": bs, "vocab_size": 151936, "num_draft_cols": 4, "draft_token_column": 0}
    for bs in (1, 8, 64, 512)
]


def native():
    """解析 draft_topk1_postprocess。"""
    try:
        from sglang.kernels.ops.speculative.topk1 import (
            draft_topk1_postprocess as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    bs, V = binding["bs"], binding["vocab_size"]
    num_draft_cols = binding["num_draft_cols"]
    draft_token_column = binding["draft_token_column"]
    g = torch.Generator(device=device).manual_seed(0)
    next_token_logits = torch.randn(bs, V, dtype=torch.float32, device=device, generator=g)
    positions = torch.randint(0, 4096, (bs,), dtype=torch.int64, device=device, generator=g)
    draft_tokens = torch.randint(
        0, V, (bs, num_draft_cols), dtype=torch.int64, device=device, generator=g
    )
    args = (next_token_logits, positions)
    kwargs = {"draft_tokens": draft_tokens, "draft_token_column": draft_token_column}
    return args, kwargs


def key_shape(binding):
    return [binding["bs"], binding["vocab_size"]]


def config(binding, dtype):
    bs, V = binding["bs"], binding["vocab_size"]
    num_draft_cols = binding["num_draft_cols"]
    draft_token_column = binding["draft_token_column"]
    return {
        "inputs": {
            "next_token_logits": {"shape": [bs, V], "dtype": "torch.float32"},
            "positions": {"shape": [bs], "dtype": "torch.int64", "note": "原地写回"},
            "draft_tokens": {"shape": [bs, num_draft_cols], "dtype": "torch.int64",
                             "note": "原地写回"},
            "draft_token_column": {"scalar": draft_token_column},
        },
        "outputs": {
            "positions": {"shape": [bs], "dtype": "torch.int64"},
            "draft_tokens": {"shape": [bs, num_draft_cols], "dtype": "torch.int64"},
        },
        "dims": {"bs": bs, "vocab_size": V, "num_draft_cols": num_draft_cols,
                 "draft_token_column": draft_token_column},
    }
