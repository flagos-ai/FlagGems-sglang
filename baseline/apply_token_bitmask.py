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

"""apply_token_bitmask baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.grammar.bitmask_ops.apply_token_bitmask_inplace_triton
    调用形态复刻 problems/sampling_grammar/apply_token_bitmask/baseline.py：
        logits = logits.clone()
        apply_token_bitmask_inplace_triton(logits, bitmask)
    native 原地改写 logits（baseline 先 clone）；故 IS_INPLACE=True，
    采样前由采集器用 build_inputs 重建实参，避免 warmup 原地累积污染。

输入构造复刻 problems/sampling_grammar/apply_token_bitmask/cases.py 的 BENCH_CASES：
    logits=randn(B, V) float32；words=(V+31)//32；
    bitmask=randint(int32 全域, (B, words)) int32。
dtype 取 benchmark/test_apply_token_bitmask.py 的 [float32]。
"""

import torch

OP_NAME = "apply_token_bitmask"
DTYPES = [torch.float32]
IS_INPLACE = True  # native 原地改写 logits

# BENCH_CASES: _case(b, 152064) for b in (1,8,64,512,4096)
_SHAPES = [{"B": b, "V": 152064} for b in (1, 8, 64, 512, 4096)]


def native():
    """解析 apply_token_bitmask_inplace_triton。"""
    try:
        from sglang.kernels.ops.grammar.bitmask_ops import (
            apply_token_bitmask_inplace_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    B, V = binding["B"], binding["V"]
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(B, V, dtype=dtype, device=device, generator=g)
    words = (V + 31) // 32
    bitmask = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (B, words),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    return (logits, bitmask), {}


def key_shape(binding):
    return [binding["B"], binding["V"]]


def config(binding, dtype):
    B, V = binding["B"], binding["V"]
    words = (V + 31) // 32
    dt = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [B, V], "dtype": dt, "note": "原地写回"},
            "bitmask": {"shape": [B, words], "dtype": "torch.int32"},
        },
        "outputs": {
            "logits": {"shape": [B, V], "dtype": dt},
        },
        "dims": {"B": B, "V": V, "words": words},
    }
