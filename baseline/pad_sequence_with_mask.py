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

"""pad_sequence_with_mask baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.pad.pad_sequence_with_mask
    调用形态复刻 problems/attention/pad_sequence_with_mask/baseline.py：
        pad_sequence_with_mask(input_emb, offsets, lengths, max_len)
    返回新张量（含 padded 序列与 mask），不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    lengths = randint(1, max_len+1, (bs,), int32)
    offsets = 前缀和（offsets[1:] = cumsum(lengths)[:-1]），total = sum(lengths)
    emb = randn(total, hidden).bf16；max_len=64, hidden=4096。seed=9。total 数据相关。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "pad_sequence_with_mask"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_MAX_LEN = 64
_HIDDEN = 4096
_CASES = [1, 8, 64, 256]


def native():
    """解析 sglang.kernels.ops.attention.pad.pad_sequence_with_mask。"""
    try:
        from sglang.kernels.ops.attention.pad import pad_sequence_with_mask as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "max_len": _MAX_LEN, "hidden": _HIDDEN} for bs in _CASES]


def build_inputs(binding, dtype, device):
    bs, max_len, hidden = binding["bs"], binding["max_len"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(9)
    lengths = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    offsets = torch.zeros(bs, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]
    total = int(lengths.sum())
    emb = torch.randn(total, hidden, device=device, generator=g).to(dtype)
    return (emb, offsets, lengths, max_len), {}


def key_shape(binding):
    return "bs%d_ml%d_h%d" % (binding["bs"], binding["max_len"], binding["hidden"])


def config(binding, dtype):
    bs, max_len, hidden = binding["bs"], binding["max_len"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "input_emb": {"shape": ["total_tokens", hidden], "dtype": dt},
            "offsets": {"shape": [bs], "dtype": "torch.int32"},
            "lengths": {"shape": [bs], "dtype": "torch.int32"},
            "max_len": {"scalar": max_len},
        },
        "outputs": {
            "padded": {"shape": [bs, max_len, hidden], "dtype": dt},
            "mask": {"shape": [bs, max_len], "note": "有效位 mask"},
        },
        "dims": {
            "bs": bs, "max_len": max_len, "hidden": hidden,
            "note": "total_tokens = sum(randint(1,max_len+1)) 数据相关（seed=9）",
        },
    }
