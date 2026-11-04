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

"""quantize_fp4_indexer baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsv4.fp4_indexer.quantize_fp4_indexer_tensor
    调用形态复刻 baseline.py：
        quantize_fp4_indexer_tensor(x)
    返回 (q_packed, sf) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x=randn(n,128).to(bf16).contiguous()。
BENCH: _case(n) for n in (16,128,1024,8192,65536)。
"""

import torch

OP_NAME = "quantize_fp4_indexer"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_D = 128
_SHAPES = [{"n": n, "d": _D} for n in (16, 128, 1024, 8192, 65536)]


def native():
    """解析 quantize_fp4_indexer_tensor。"""
    try:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    n, d = binding["n"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(n, d, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (x.contiguous(),), {}


def key_shape(binding):
    return [binding["n"], binding["d"]]


def config(binding, dtype):
    n, d = binding["n"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [n, d], "dtype": dt},
        },
        "outputs": {
            "q_packed": {"shape": [n, d // 2], "dtype": "torch.uint8",
                         "note": "fp4 每 2 个打包进 1 字节"},
            "sf": {"shape": [n, d // 32], "dtype": "torch.uint8",
                   "note": "UE8M0 group scale"},
        },
        "dims": {"n": n, "d": d},
    }
