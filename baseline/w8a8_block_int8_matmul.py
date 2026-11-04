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

"""w8a8_block_int8_matmul baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.quantization.int8_kernel.w8a8_block_int8_matmul
    调用形态复刻 problems/quantization/w8a8_block_int8_matmul/baseline.py：
        w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype=output_dtype)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/quantization/w8a8_block_int8_matmul/cases.py 的 BENCH_CASES：
    block=[128,128]；A=randint(-8,8,(m,k)) int8；B=randint(-8,8,(n,k)) int8；
    As=1e-2*rand(m, k//128)；Bs=1e-2*rand(n//128, k//128)；output_dtype=dtype。
dtype 取 benchmark/test_w8a8_block_int8_matmul.py 的 [bfloat16]。
"""

import torch

OP_NAME = "w8a8_block_int8_matmul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_BLOCK = [128, 128]
# BENCH_CASES: _case(m,n,k) for m in (1,8,64,512,4096)
#   for n,k in ((1024,4096),(4096,4096),(7168,4096))
_SHAPES = [
    {"m": m, "n": n, "k": k}
    for m in (1, 8, 64, 512, 4096)
    for (n, k) in ((1024, 4096), (4096, 4096), (7168, 4096))
]


def native():
    """解析 w8a8_block_int8_matmul。

    spec/baseline 坐标为 quantization.int8_kernel；sglang 0.5.21 把该符号挪到了
    gemm.int8_kernel，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.quantization.int8_kernel import (
            w8a8_block_int8_matmul as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.gemm.int8_kernel import (
                w8a8_block_int8_matmul as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    block_n, block_k = _BLOCK
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=device, generator=g)
    B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device, generator=g)
    As = (
        1e-2 * torch.rand(m, k // block_k, dtype=torch.float32, device=device, generator=g)
    ).contiguous()
    Bs = (
        1e-2
        * torch.rand(
            n // block_n, k // block_k, dtype=torch.float32, device=device, generator=g
        )
    ).contiguous()
    return (A, B, As, Bs, _BLOCK), {"output_dtype": dtype}


def key_shape(binding):
    return [binding["m"], binding["n"], binding["k"]]


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    block_n, block_k = _BLOCK
    dt = str(dtype)
    return {
        "inputs": {
            "A": {"shape": [m, k], "dtype": "torch.int8"},
            "B": {"shape": [n, k], "dtype": "torch.int8"},
            "As": {"shape": [m, k // block_k], "dtype": "torch.float32"},
            "Bs": {"shape": [n // block_n, k // block_k], "dtype": "torch.float32"},
            "block_size": {"scalar": list(_BLOCK)},
            "output_dtype": {"scalar": dt},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n, "k": k, "block_n": block_n, "block_k": block_k},
    }
