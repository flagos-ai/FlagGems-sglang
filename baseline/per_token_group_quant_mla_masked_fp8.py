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

"""per_token_group_quant_mla_masked_fp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_kernel.
        per_token_group_quant_mla_deep_gemm_masked_fp8
    调用形态复刻 baseline.py：
        per_token_group_quant_mla_deep_gemm_masked_fp8(x, group_size=group_size, eps=eps)
    返回 (x_q, x_s, masked_m, m, aligned_m)，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（group_size=128, eps=1e-12）：
    x=randn(b,m,k).to(bf16).contiguous()。
BENCH: _case(b,m,512) for b in (1,8,32) for m in (16,128,1024)。
"""

import torch

OP_NAME = "per_token_group_quant_mla_masked_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_GROUP = 128
_EPS = 1e-12
_K = 512
_SHAPES = [
    {"b": b, "m": m, "k": _K}
    for b in (1, 8, 32)
    for m in (16, 128, 1024)
]


def native():
    """解析 per_token_group_quant_mla_deep_gemm_masked_fp8。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_mla_deep_gemm_masked_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    b, m, k = binding["b"], binding["m"], binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(b, m, k, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (x.contiguous(),), {"group_size": _GROUP, "eps": _EPS}


def key_shape(binding):
    return f"b{binding['b']}_m{binding['m']}_k{binding['k']}"


def config(binding, dtype):
    b, m, k = binding["b"], binding["m"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [b, m, k], "dtype": dt},
            "group_size": {"scalar": _GROUP},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "x_q": {"shape": [b, "aligned_m", k], "dtype": "torch.float8_e4m3fn",
                    "note": "aligned_m 为 m 对齐后的上界，数据相关"},
            "x_s": {"shape": [b, "aligned_m", k // _GROUP], "dtype": "torch.float32"},
            "masked_m": {"shape": [b], "dtype": "torch.int32"},
            "m": {"note": "标量，本 case = m"},
            "aligned_m": {"note": "标量，m 对齐后的上界"},
        },
        "dims": {"b": b, "m": m, "k": k, "group_size": _GROUP},
    }
