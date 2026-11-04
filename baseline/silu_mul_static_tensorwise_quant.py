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

"""silu_mul_static_tensorwise_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.
silu_mul_static_tensorwise_quant_for_cutlass_moe（destination-passing）。复刻
problems/moe/silu_mul_static_tensorwise_quant/baseline.py：out=output.clone()，
调 op(input, out, scale, num_tokens_tensor, expected_num_tokens,
intermediate_size)，原地写 out。scale 为静态输入，不被写回。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, max_tokens=t, intermediate=2048,
输入 dtype=bf16）：input=randn(max,2*inter).bf16; output=zeros(max,inter).fp8_e4m3;
scale=tensor([4.0]).fp32; num_tokens_tensor=tensor([t]).int32。IS_INPLACE=True。
"""

import torch

OP_NAME = "silu_mul_static_tensorwise_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(t, t, 2048) for t in (1, 8, 64, 512, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_mul_static_tensorwise_quant_for_cutlass_moe as _f,
        )
    except Exception:
        return None

    def _op(input, out, scale, num_tokens_tensor, expected_num_tokens, intermediate_size):
        _f(input, out, scale, num_tokens_tensor, expected_num_tokens, intermediate_size)
        return out

    return _op


def grid():
    return [{"num_tokens": t, "max_tokens": mt, "inter": i} for (t, mt, i) in _SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, max_tokens, inter = (
        binding["num_tokens"], binding["max_tokens"], binding["inter"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(max_tokens, 2 * inter, device=device, generator=g).to(dtype)
    output = torch.zeros(max_tokens, inter, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.tensor([4.0], dtype=torch.float32, device=device)
    num_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    return (x, output, scale, num_tokens_tensor, max_tokens, inter), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_mt{binding['max_tokens']}_inter{binding['inter']}"


def config(binding, dtype):
    num_tokens, max_tokens, inter = (
        binding["num_tokens"], binding["max_tokens"], binding["inter"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [max_tokens, 2 * inter], "dtype": dt},
            "output": {"shape": [max_tokens, inter], "dtype": "torch.float8_e4m3fn", "note": "原地写回（output.clone()）"},
            "scale": {"shape": [1], "dtype": "torch.float32"},
            "num_tokens_tensor": {"shape": [1], "dtype": "torch.int32"},
            "expected_num_tokens": {"scalar": max_tokens},
            "intermediate_size": {"scalar": inter},
        },
        "outputs": {
            "output": {"shape": [max_tokens, inter], "dtype": "torch.float8_e4m3fn"},
        },
        "dims": {"num_tokens": num_tokens, "max_tokens": max_tokens, "intermediate_size": inter},
    }
