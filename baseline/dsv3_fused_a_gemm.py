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

"""dsv3_fused_a_gemm baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.gemm.dsv3_fused_a_gemm.dsv3_fused_a_gemm
    调用形态复刻 problems/gemm/dsv3_fused_a_gemm/baseline.py：
        dsv3_fused_a_gemm(mat_a, mat_b)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/gemm/dsv3_fused_a_gemm/cases.py 的 BENCH_CASES：
    a = randn(num_tokens, hd_in).bf16
    mat_b = randn(hd_out, hd_in).bf16 的转置（列主序 [hd_in, hd_out]）
    其中 hd_in=7168, hd_out=2112。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "dsv3_fused_a_gemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: num_tokens in (1,2,4,8,16); hd_in=7168, hd_out=2112
_HD_IN = 7168
_HD_OUT = 2112
_TOKENS = [1, 2, 4, 8, 16]


def native():
    """解析 sglang.kernels.ops.gemm.dsv3_fused_a_gemm.dsv3_fused_a_gemm。"""
    try:
        from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"num_tokens": m, "hd_in": _HD_IN, "hd_out": _HD_OUT} for m in _TOKENS]


def build_inputs(binding, dtype, device):
    m, hd_in, hd_out = binding["num_tokens"], binding["hd_in"], binding["hd_out"]
    g = torch.Generator(device=device).manual_seed(9)
    a = torch.randn(m, hd_in, device=device, generator=g).to(dtype)
    w = torch.randn(hd_out, hd_in, device=device, generator=g).to(dtype)
    return (a, w.t()), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["hd_in"], binding["hd_out"]]


def config(binding, dtype):
    m, hd_in, hd_out = binding["num_tokens"], binding["hd_in"], binding["hd_out"]
    dt = str(dtype)
    return {
        "inputs": {
            "mat_a": {"shape": [m, hd_in], "dtype": dt},
            "mat_b": {"shape": [hd_in, hd_out], "dtype": dt,
                      "note": "列主序（[hd_out, hd_in] 的转置视图）"},
        },
        "outputs": {
            "out": {"shape": [m, hd_out], "dtype": dt},
        },
        "dims": {"num_tokens": m, "hd_in": hd_in, "hd_out": hd_out},
    }
