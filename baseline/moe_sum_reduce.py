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

"""moe_sum_reduce baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.fused_moe_triton_kernels.moe_sum_reduce_triton
    调用形态复刻 problems/moe/moe_sum_reduce/baseline.py：
        output = empty(T, H); moe_sum_reduce_triton(input, output, routed_scaling_factor)
    kernel 把结果原地写入 output -> IS_INPLACE=True（采样前用 build_inputs
    重建实参，避免 warmup 的原地累积污染）。

输入构造复刻 problems/moe/moe_sum_reduce/cases.py 的 BENCH_CASES：
    input=randn(T, top_k, H).to(dtype); routed_scaling_factor=2.5
    top_k=8, H=7168。
dtype 取 benchmark/test_moe_sum_reduce.py 的 [bfloat16]。
"""

import torch

OP_NAME = "moe_sum_reduce"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 output

_SCALE = 2.5
_SHAPES = [(t, 8, 7168) for t in (1, 8, 64, 512, 4096)]


def native():
    """解析 moe_sum_reduce_triton。"""
    try:
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            moe_sum_reduce_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"T": t, "top_k": tk, "H": h} for (t, tk, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    T, top_k, H = binding["T"], binding["top_k"], binding["H"]
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(
        T, top_k, H, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    output = torch.empty(T, H, dtype=inp.dtype, device=inp.device)
    return (inp, output, _SCALE), {}


def key_shape(binding):
    return [binding["T"], binding["top_k"], binding["H"]]


def config(binding, dtype):
    T, top_k, H = binding["T"], binding["top_k"], binding["H"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [T, top_k, H], "dtype": dt},
            "output": {"shape": [T, H], "dtype": dt, "note": "原地写回"},
            "routed_scaling_factor": {"scalar": _SCALE},
        },
        "outputs": {
            "output": {"shape": [T, H], "dtype": dt},
        },
        "dims": {"T": T, "top_k": top_k, "H": H},
    }
