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

"""silu_and_mul_masked baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.ep_moe_kernels.silu_and_mul_masked_fwd
    调用形态复刻 problems/moe/silu_and_mul_masked/baseline.py：
        output = empty(E, T, H // 2, dtype=bfloat16);
        silu_and_mul_masked_fwd(input, output, masked_m)
    kernel 把结果原地写入 output -> IS_INPLACE=True（采样前用 build_inputs
    重建实参，避免 warmup 的原地累积污染）。

输入构造复刻 problems/moe/silu_and_mul_masked/cases.py 的 BENCH_CASES：
    input=randn(expert_num, token_num_padded, hidden_dim).to(dtype);
    masked_m=tensor([256]*e) int32
    token_num_padded=256, hidden_dim=4096, expert_num ∈ {4, 8, 32}。
dtype 取 benchmark/test_silu_and_mul_masked.py 的 [bfloat16]。
"""

import torch

OP_NAME = "silu_and_mul_masked"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 output

# (expert_num, token_num_padded, hidden_dim)
_SHAPES = [(e, 256, 4096) for e in (4, 8, 32)]


def native():
    """解析 silu_and_mul_masked_fwd。"""
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"E": e, "T": t, "H": h} for (e, t, h) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    E, T, H = binding["E"], binding["T"], binding["H"]
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(
        E, T, H, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    masked_m = torch.tensor([T] * E, dtype=torch.int32, device=device)
    output = torch.empty(E, T, H // 2, dtype=torch.bfloat16, device=inp.device)
    return (inp, output, masked_m), {}


def key_shape(binding):
    return [binding["E"], binding["T"], binding["H"]]


def config(binding, dtype):
    E, T, H = binding["E"], binding["T"], binding["H"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [E, T, H], "dtype": dt},
            "output": {"shape": [E, T, H // 2], "dtype": "torch.bfloat16",
                       "note": "原地写回"},
            "masked_m": {"shape": [E], "dtype": "torch.int32"},
        },
        "outputs": {
            "output": {"shape": [E, T, H // 2], "dtype": "torch.bfloat16"},
        },
        "dims": {"E": E, "T": T, "H": H},
    }
