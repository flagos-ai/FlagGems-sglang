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

"""triton_scaled_mm baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_kernel.triton_scaled_mm
    调用形态复刻 baseline.py：
        triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（in_dtype=fp8_e4m3, out_dtype=fp16, bias=None）：
    a=clamp(0.1*randn(m,k),-0.3,0.3).to(fp8)；b=clamp(0.1*randn(k,n),...).to(fp8)；
    scale_a=0.1+0.05*rand(m,1) fp32；scale_b=0.1+0.05*rand(n,1) fp32。
BENCH: _case(m,k,n,fp8) for m in (1,8,64,512,4096) for k,n in ((4096,4096),(4096,7168))。
DTYPES 取输入 fp8_e4m3fn。
"""

import torch

OP_NAME = "triton_scaled_mm"
DTYPES = [torch.float8_e4m3fn]
IS_INPLACE = False

_OUT_DTYPE = torch.float16
_CASES = [
    (m, k, n)
    for m in (1, 8, 64, 512, 4096)
    for (k, n) in ((4096, 4096), (4096, 7168))
]


def native():
    """解析 triton_scaled_mm。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.gemm.fp8_kernel，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            triton_scaled_mm as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.gemm.fp8_kernel import (
                triton_scaled_mm as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "k": k, "n": n} for (m, k, n) in _CASES]


def build_inputs(binding, dtype, device):
    m, k, n = binding["m"], binding["k"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.clamp(
        0.1 * torch.randn(m, k, dtype=torch.float16, device=device, generator=g),
        -0.3, 0.3,
    ).to(dtype)
    b = torch.clamp(
        0.1 * torch.randn(k, n, dtype=torch.float16, device=device, generator=g),
        -0.3, 0.3,
    ).to(dtype)
    scale_a = 0.1 + 0.05 * torch.rand(
        m, 1, dtype=torch.float32, device=device, generator=g
    )
    scale_b = 0.1 + 0.05 * torch.rand(
        n, 1, dtype=torch.float32, device=device, generator=g
    )
    return (a, b, scale_a, scale_b, _OUT_DTYPE), {"bias": None}


def key_shape(binding):
    return [binding["m"], binding["k"], binding["n"]]


def config(binding, dtype):
    m, k, n = binding["m"], binding["k"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [m, k], "dtype": dt},
            "weight": {"shape": [k, n], "dtype": dt},
            "scale_a": {"shape": [m, 1], "dtype": "torch.float32"},
            "scale_b": {"shape": [n, 1], "dtype": "torch.float32"},
            "out_dtype": {"scalar": str(_OUT_DTYPE)},
            "bias": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": str(_OUT_DTYPE)},
        },
        "dims": {"m": m, "k": k, "n": n},
    }
