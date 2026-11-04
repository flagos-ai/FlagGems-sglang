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

"""dcp_lse_combine baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dcp_kernels.dcp_lse_combine_triton
    调用形态复刻 problems/attention/dcp_lse_combine/baseline.py：
        dcp_lse_combine_triton(recv_output, recv_lse,
                               is_lse_base_on_e=..., return_lse=...)
    返回新张量（output, lse），不原地写回；IS_INPLACE=False。

输入构造复刻 problems/attention/dcp_lse_combine/cases.py 的 BENCH_CASES：
    recv_output = randn(n, b, h, d).bf16；recv_lse = randn(n, b, h) fp32 * 4
    h=16, d=128，is_lse_base_on_e=True，return_lse=True。seed=0。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "dcp_lse_combine"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_H = 16
_D = 128
# BENCH_CASES: (n, b) for n in (2,4,8) for b in (1,64,1024)
_CASES = [(n, b) for n in (2, 4, 8) for b in (1, 64, 1024)]


def native():
    """解析 sglang.kernels.ops.attention.dcp_kernels.dcp_lse_combine_triton。"""
    try:
        from sglang.kernels.ops.attention.dcp_kernels import (
            dcp_lse_combine_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"n": n, "b": b, "h": _H, "d": _D} for (n, b) in _CASES]


def build_inputs(binding, dtype, device):
    n, b, h, d = binding["n"], binding["b"], binding["h"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    recv_output = torch.randn(
        n, b, h, d, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    recv_lse = (
        torch.randn(n, b, h, dtype=torch.float32, device=device, generator=g) * 4
    ).contiguous()
    return (recv_output, recv_lse), {"is_lse_base_on_e": True, "return_lse": True}


def key_shape(binding):
    return [binding["n"], binding["b"], binding["h"], binding["d"]]


def config(binding, dtype):
    n, b, h, d = binding["n"], binding["b"], binding["h"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "recv_output": {"shape": [n, b, h, d], "dtype": dt},
            "recv_lse": {"shape": [n, b, h], "dtype": "torch.float32"},
            "is_lse_base_on_e": {"scalar": True},
            "return_lse": {"scalar": True},
        },
        "outputs": {
            "output": {"shape": [b, h, d], "dtype": dt},
            "lse": {"shape": [b, h], "dtype": "torch.float32"},
        },
        "dims": {"n": n, "b": b, "h": h, "d": d},
    }
