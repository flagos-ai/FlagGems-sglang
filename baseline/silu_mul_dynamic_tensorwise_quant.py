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

"""silu_mul_dynamic_tensorwise_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.
silu_mul_dynamic_tensorwise_quant_for_cutlass_moe（destination-passing）。复刻
problems/moe/silu_mul_dynamic_tensorwise_quant/baseline.py：output/scale 各 clone，
调 op(input, output, scale, num_tokens_tensor, expected_num_tokens,
intermediate_size)，原地写 output 与 scale。

输入复刻 cases.py BENCH_CASES（t∈{64,512,4096}, live=t, inter=2048,
输入 dtype=bf16）：input=randn(t,2*inter).bf16; output=zeros(t,inter).fp8_e4m3;
scale=zeros(1).fp32; num_tokens_tensor=tensor([live]).int32。IS_INPLACE=True。
"""

import torch

OP_NAME = "silu_mul_dynamic_tensorwise_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(t, t, 2048) for t in (64, 512, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_mul_dynamic_tensorwise_quant_for_cutlass_moe as _f,
        )
    except Exception:
        return None

    def _op(input, output, scale, num_tokens_tensor, expected_num_tokens, intermediate_size):
        _f(input, output, scale, num_tokens_tensor, expected_num_tokens, intermediate_size)
        return output, scale

    return _op


def grid():
    return [{"expected_tokens": e, "live_tokens": lv, "inter": i} for (e, lv, i) in _SHAPES]


def build_inputs(binding, dtype, device):
    expected_tokens, live_tokens, inter = (
        binding["expected_tokens"], binding["live_tokens"], binding["inter"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(
        expected_tokens, 2 * inter, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    output = torch.zeros((expected_tokens, inter), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.zeros(1, dtype=torch.float32, device=device)
    num_tokens_tensor = torch.tensor([live_tokens], dtype=torch.int32, device=device)
    return (inp, output, scale, num_tokens_tensor, expected_tokens, inter), {}


def key_shape(binding):
    return f"et{binding['expected_tokens']}_lt{binding['live_tokens']}_inter{binding['inter']}"


def config(binding, dtype):
    expected_tokens, live_tokens, inter = (
        binding["expected_tokens"], binding["live_tokens"], binding["inter"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [expected_tokens, 2 * inter], "dtype": dt},
            "output": {"shape": [expected_tokens, inter], "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "scale": {"shape": [1], "dtype": "torch.float32", "note": "原地写回"},
            "num_tokens_tensor": {"shape": [1], "dtype": "torch.int32"},
            "expected_num_tokens": {"scalar": expected_tokens},
            "intermediate_size": {"scalar": inter},
        },
        "outputs": {
            "output": {"shape": [expected_tokens, inter], "dtype": "torch.float8_e4m3fn"},
            "scale": {"shape": [1], "dtype": "torch.float32"},
        },
        "dims": {"expected_tokens": expected_tokens, "live_tokens": live_tokens,
                 "intermediate_size": inter},
    }
