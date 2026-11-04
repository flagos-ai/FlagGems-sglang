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

"""awq_gemm baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.awq_triton.awq_gemm_triton
    调用形态复刻 problems/quantization/awq_gemm/baseline.py：
        awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/quantization/awq_gemm/cases.py 的 BENCH_CASES：
    group_size=128, split_k=8, k=4096；x=rand fp32 (m,k)；
    qweight=randint int32 (k, n//8)；qzeros=randint int32 (k//gs, n//8)；
    scales=rand fp32 (k//gs, n)。
BENCH: _case(m,4096,n,128,8) for m in (1,8,64,512) for n in (1024,4096)。
"""

import torch

OP_NAME = "awq_gemm"
DTYPES = [torch.float32]
IS_INPLACE = False

_GROUP = 128
_SPLIT_K = 8
_K = 4096
_SHAPES = [
    {"m": m, "k": _K, "n": n}
    for m in (1, 8, 64, 512)
    for n in (1024, 4096)
]


def native():
    """解析 awq_gemm_triton。"""
    try:
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_gemm_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, k, n = binding["m"], binding["k"], binding["n"]
    gs = _GROUP
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand(m, k, dtype=torch.float32, device=device, generator=g)
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (k, n // 8), dtype=torch.int32,
        device=device, generator=g,
    )
    qzeros = torch.randint(
        0, torch.iinfo(torch.int32).max, (k // gs, n // 8), dtype=torch.int32,
        device=device, generator=g,
    )
    scales = torch.rand(k // gs, n, dtype=torch.float32, device=device, generator=g)
    return (x, qweight, scales, qzeros, _SPLIT_K), {}


def key_shape(binding):
    return [binding["m"], binding["k"], binding["n"]]


def config(binding, dtype):
    m, k, n = binding["m"], binding["k"], binding["n"]
    gs = _GROUP
    return {
        "inputs": {
            "input": {"shape": [m, k], "dtype": "torch.float32"},
            "qweight": {"shape": [k, n // 8], "dtype": "torch.int32"},
            "scales": {"shape": [k // gs, n], "dtype": "torch.float32"},
            "qzeros": {"shape": [k // gs, n // 8], "dtype": "torch.int32"},
            "split_k_iters": {"scalar": _SPLIT_K},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "k": k, "n": n, "group_size": gs, "split_k": _SPLIT_K},
    }
