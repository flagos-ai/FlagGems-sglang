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

"""vocab_parallel_embedding baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.embeddings.vocab_parallel_embedding.vocab_parallel_embedding
    调用形态复刻 problems/utility/vocab_parallel_embedding/baseline.py：
        vocab_parallel_embedding(input_, weight, org_vocab_start_index,
            org_vocab_end_index, num_org_vocab_padding,
            added_vocab_start_index, added_vocab_end_index)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/utility/vocab_parallel_embedding/cases.py 的 BENCH_CASES：
    org=[1000,5000], padding=8, added=[20000,20064]
    local_vocab = (org_end-org_start)+padding+(added_end-added_start)
    weight = randn(local_vocab, hidden).bf16
    tokens = randint(0,30000,(num_tokens,)) int32，部分改到 org / added 窗口内。
    其中 num_tokens in (1,8,64,512,4096)，hidden in (2048, 7168)。
dtype 取 cases.py 的 [bfloat16]（weight dtype；tokens 为 int32）。
"""

import torch

OP_NAME = "vocab_parallel_embedding"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_ORG_START, _ORG_END = 1000, 5000
_PADDING = 8
_ADDED_START, _ADDED_END = 20000, 20064
_TOKENS = [1, 8, 64, 512, 4096]
_HIDDEN = [2048, 7168]


def native():
    """解析 vocab_parallel_embedding。"""
    try:
        from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
            vocab_parallel_embedding as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"num_tokens": t, "hidden": h} for t in _TOKENS for h in _HIDDEN]


def _local_vocab():
    return (_ORG_END - _ORG_START) + _PADDING + (_ADDED_END - _ADDED_START)


def build_inputs(binding, dtype, device):
    num_tokens, hidden = binding["num_tokens"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(9)
    local_vocab = _local_vocab()
    weight = torch.randn(local_vocab, hidden, device=device, generator=g).to(dtype)
    tokens = torch.randint(
        0, 30000, (num_tokens,), dtype=torch.int32, device=device, generator=g
    )
    tokens[::3] = torch.randint(
        _ORG_START, _ORG_END, (tokens[::3].numel(),), dtype=torch.int32,
        device=device, generator=g,
    )
    tokens[1::3] = torch.randint(
        _ADDED_START, _ADDED_END, (tokens[1::3].numel(),), dtype=torch.int32,
        device=device, generator=g,
    )
    return (
        tokens.contiguous(), weight,
        _ORG_START, _ORG_END, _PADDING, _ADDED_START, _ADDED_END,
    ), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["hidden"]]


def config(binding, dtype):
    num_tokens, hidden = binding["num_tokens"], binding["hidden"]
    dt = str(dtype)
    local_vocab = _local_vocab()
    return {
        "inputs": {
            "input_": {"shape": [num_tokens], "dtype": "torch.int32"},
            "weight": {"shape": [local_vocab, hidden], "dtype": dt},
            "org_vocab_start_index": {"scalar": _ORG_START},
            "org_vocab_end_index": {"scalar": _ORG_END},
            "num_org_vocab_padding": {"scalar": _PADDING},
            "added_vocab_start_index": {"scalar": _ADDED_START},
            "added_vocab_end_index": {"scalar": _ADDED_END},
        },
        "outputs": {
            "out": {"shape": [num_tokens, hidden], "dtype": dt},
        },
        "dims": {
            "num_tokens": num_tokens, "hidden": hidden, "local_vocab": local_vocab,
            "org_vocab": [_ORG_START, _ORG_END], "padding": _PADDING,
            "added_vocab": [_ADDED_START, _ADDED_END],
        },
    }
