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

"""fp8_per_token_to_per_tensor_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.
fp8_per_token_to_per_tensor_quant_triton。复刻 problems/moe/
fp8_per_token_to_per_tensor_quant/baseline.py：output=empty_like(x)，调
op(x, x_scale, masked_m, output_scale, output)，原地写 output。

输入复刻 cases.py BENCH_CASES（e∈{4,8,32}, token_num_padded=256, k=4096,
counts=[256]*e）：x=(randn(e,tp,k)*2).fp8_e4m3; x_scale=rand(e,tp,k//128).fp32;
masked_m=tensor(counts).int32; output_scale=tensor([0.02]).fp32。
IS_INPLACE=True（写回 output 缓冲）。
"""

import torch

OP_NAME = "fp8_per_token_to_per_tensor_quant"
DTYPES = [torch.float8_e4m3fn]
IS_INPLACE = True

_K_SCALE_BLOCK = 128
_SHAPES = [(e, 256, 4096) for e in (4, 8, 32)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            fp8_per_token_to_per_tensor_quant_triton as _f,
        )
    except Exception:
        return None

    def _op(x, x_scale, masked_m, output_scale, output):
        _f(x, x_scale, masked_m, output_scale, output)
        return output

    return _op


def grid():
    return [{"expert_num": e, "token_num_padded": tp, "k": k} for (e, tp, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    e, tp, k = binding["expert_num"], binding["token_num_padded"], binding["k"]
    num_groups = k // _K_SCALE_BLOCK
    g = torch.Generator(device=device).manual_seed(0)
    x = (torch.randn(e, tp, k, generator=g, device=device) * 2).to(dtype)
    x_scale = torch.rand(
        e, tp, num_groups, generator=g, device=device, dtype=torch.float32
    ) * 0.1 + 0.01
    masked_m = torch.tensor([tp] * e, dtype=torch.int32, device=device)
    output_scale = torch.tensor([0.02], dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    return (x, x_scale, masked_m, output_scale, output), {}


def key_shape(binding):
    return f"e{binding['expert_num']}_tp{binding['token_num_padded']}_k{binding['k']}"


def config(binding, dtype):
    e, tp, k = binding["expert_num"], binding["token_num_padded"], binding["k"]
    dt = str(dtype)
    num_groups = k // _K_SCALE_BLOCK
    return {
        "inputs": {
            "x": {"shape": [e, tp, k], "dtype": dt},
            "x_scale": {"shape": [e, tp, num_groups], "dtype": "torch.float32"},
            "masked_m": {"shape": [e], "dtype": "torch.int32"},
            "output_scale": {"shape": [1], "dtype": "torch.float32"},
            "output": {"shape": [e, tp, k], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "output": {"shape": [e, tp, k], "dtype": dt},
        },
        "dims": {"expert_num": e, "token_num_padded": tp, "k": k, "num_groups": num_groups},
    }
