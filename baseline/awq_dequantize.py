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

"""awq_dequantize baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.awq_triton.awq_dequantize_triton
    调用形态复刻 problems/quantization/awq_dequantize/baseline.py：
        awq_dequantize_triton(qweight, scales, zeros)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/quantization/awq_dequantize/cases.py 的 BENCH_CASES：
    group_size=128；qweight=randint int32 (rows, cols)；
    scales=rand fp16 (rows//gs, cols*8)；zeros=randint int32 (rows//gs, cols)。
BENCH: _case(rows, cols, 128) for rows in (3584,18944) for cols in (448,4736)。
dtype（cases 里量化张量为 int32 / scales fp16）故 DTYPES 取 [float16]。
"""

import torch

OP_NAME = "awq_dequantize"
DTYPES = [torch.float16]
IS_INPLACE = False

_GROUP = 128
_SHAPES = [
    {"rows": rows, "cols": cols}
    for rows in (3584, 18944)
    for cols in (448, 4736)
]


def native():
    """解析 awq_dequantize_triton。"""
    try:
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    rows, cols = binding["rows"], binding["cols"]
    gs = _GROUP
    g = torch.Generator(device=device).manual_seed(0)
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (rows, cols), dtype=torch.int32,
        device=device, generator=g,
    )
    scales = torch.rand(
        rows // gs, cols * 8, dtype=torch.float16, device=device, generator=g
    )
    zeros = torch.randint(
        0, torch.iinfo(torch.int32).max, (rows // gs, cols), dtype=torch.int32,
        device=device, generator=g,
    )
    return (qweight, scales, zeros), {}


def key_shape(binding):
    return [binding["rows"], binding["cols"]]


def config(binding, dtype):
    rows, cols = binding["rows"], binding["cols"]
    gs = _GROUP
    return {
        "inputs": {
            "qweight": {"shape": [rows, cols], "dtype": "torch.int32"},
            "scales": {"shape": [rows // gs, cols * 8], "dtype": "torch.float16"},
            "zeros": {"shape": [rows // gs, cols], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [rows, cols * 8], "dtype": "torch.float16"},
        },
        "dims": {"rows": rows, "cols": cols, "group_size": gs},
    }
