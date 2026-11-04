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

"""fill_bonus_tokens baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.eagle.fill_bonus_tokens_func
    调用形态复刻 problems/speculative/fill_bonus_tokens/baseline.py：
        out = bonus_tokens.clone()
        fill_bonus_tokens_func(accept_tokens, accept_lens, out,
                               accept_stride, accept_lens.shape[0])
    kernel 原地写第三个实参 out -> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES：
    accept_tokens = randint(0, 128000, (batch, width)) int64
    accept_lens = randint(1, width+1, (batch,)) int64
    bonus_tokens = zeros(batch) int64; accept_stride=width
    其中 width=8，batch in (1,8,64,512,4096)。
全整型，DTYPES 记 int64。
"""

import torch

OP_NAME = "fill_bonus_tokens"
DTYPES = [torch.int64]
IS_INPLACE = True  # kernel 原地写 bonus_tokens

_WIDTH = 8
_BATCH = [1, 8, 64, 512, 4096]


def native():
    """解析 fill_bonus_tokens_func。"""
    try:
        from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"batch": b, "width": _WIDTH} for b in _BATCH]


def build_inputs(binding, dtype, device):
    batch, width = binding["batch"], binding["width"]
    g = torch.Generator(device=device).manual_seed(9)
    tokens = torch.randint(
        0, 128000, (batch, width), dtype=torch.int64, device=device, generator=g
    )
    lens = torch.randint(
        1, width + 1, (batch,), dtype=torch.int64, device=device, generator=g
    )
    bonus_tokens = torch.zeros(batch, dtype=torch.int64, device=device)
    return (tokens, lens, bonus_tokens, width, lens.shape[0]), {}


def key_shape(binding):
    return [binding["batch"], binding["width"]]


def config(binding, dtype):
    batch, width = binding["batch"], binding["width"]
    return {
        "inputs": {
            "accept_tokens": {"shape": [batch, width], "dtype": "torch.int64"},
            "accept_lens": {"shape": [batch], "dtype": "torch.int64"},
            "bonus_tokens": {"shape": [batch], "dtype": "torch.int64",
                             "note": "原地写回"},
            "accept_stride": {"scalar": width},
            "batch": {"scalar": batch},
        },
        "outputs": {
            "bonus_tokens": {"shape": [batch], "dtype": "torch.int64"},
        },
        "dims": {"batch": batch, "width": width},
    }
