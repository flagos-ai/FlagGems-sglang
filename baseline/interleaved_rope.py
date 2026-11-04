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

"""interleaved_rope baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.mrope.apply_interleaved_rope_triton
    调用形态复刻 problems/rope/interleaved_rope/baseline.py：
        apply_interleaved_rope_triton(x, mrope_section)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/rope/interleaved_rope/cases.py 的 BENCH_CASES：
    x = randn(3, s, d).to(dtype)；mrope_section=[16,24,24]。
dtype 取 benchmark/test_interleaved_rope.py 的 [bfloat16]。
"""

import torch

OP_NAME = "interleaved_rope"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: _case(s, 128, [16, 24, 24]) for s in (1, 128, 2048, 8192)
_SHAPES = [{"s": s, "d": 128, "mrope_section": [16, 24, 24]} for s in (1, 128, 2048, 8192)]


def native():
    """解析 apply_interleaved_rope_triton。"""
    try:
        from sglang.kernels.ops.attention.mrope import (
            apply_interleaved_rope_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    s, d = binding["s"], binding["d"]
    mrope_section = binding["mrope_section"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(3, s, d, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (x, mrope_section), {}


def key_shape(binding):
    return [3, binding["s"], binding["d"]]


def config(binding, dtype):
    s, d = binding["s"], binding["d"]
    mrope_section = binding["mrope_section"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [3, s, d], "dtype": dt},
            "mrope_section": {"scalar": list(mrope_section)},
        },
        "outputs": {
            "out": {"shape": [3, s, d], "dtype": dt},
        },
        "dims": {"s": s, "d": d, "mrope_section": list(mrope_section)},
    }
