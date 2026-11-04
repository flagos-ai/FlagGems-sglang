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

"""silu_and_mul_masked_post_per_tensor_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.
silu_and_mul_masked_post_per_tensor_quant_fwd。复刻 problems/moe/
silu_and_mul_masked_post_per_tensor_quant/baseline.py：H=H2//2，
output=empty(E,T,H,fp8_e4m3)，调 op(input, output, masked_m, scale)，原地写 output。

输入复刻 cases.py BENCH_CASES（e∈{4,8,32}, token_num_padded=256, hidden_dim=4096,
counts=[256]*e, 输入 dtype=bf16）：input=randn(E,T,2H).bf16;
masked_m=tensor(counts).int32; scale=tensor([0.5]).fp32。IS_INPLACE=True（写回 output）。
"""

import torch

OP_NAME = "silu_and_mul_masked_post_per_tensor_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(e, 256, 4096) for e in (4, 8, 32)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_per_tensor_quant_fwd as _f,
        )
    except Exception:
        return None

    def _op(input, output, masked_m, scale):
        _f(input, output, masked_m, scale)
        return output

    return _op


def grid():
    return [{"E": e, "T": t, "H": h} for (e, t, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    E, T, H = binding["E"], binding["T"], binding["H"]
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(E, T, H * 2, generator=g, device=device, dtype=torch.float32).to(dtype)
    masked_m = torch.tensor([T] * E, dtype=torch.int32, device=device)
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    output = torch.empty(E, T, H, dtype=torch.float8_e4m3fn, device=device)
    return (inp, output, masked_m, scale), {}


def key_shape(binding):
    return f"E{binding['E']}_T{binding['T']}_H{binding['H']}"


def config(binding, dtype):
    E, T, H = binding["E"], binding["T"], binding["H"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [E, T, H * 2], "dtype": dt},
            "output": {"shape": [E, T, H], "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "masked_m": {"shape": [E], "dtype": "torch.int32"},
            "scale": {"shape": [1], "dtype": "torch.float32"},
        },
        "outputs": {
            "output": {"shape": [E, T, H], "dtype": "torch.float8_e4m3fn"},
        },
        "dims": {"E": E, "T": T, "H": H, "H2": H * 2},
    }
