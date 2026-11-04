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

"""silu_and_mul_masked_post_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.silu_and_mul_masked_post_quant_fwd。
复刻 problems/moe/silu_and_mul_masked_post_quant/baseline.py：H=H2//2, G=H//group，
output=empty(E,T,H,fp8_e4m3)，output_scale=empty(E,T,G,fp32)，调
op(input, output, output_scale, quant_group_size, masked_m)，原地写 output/output_scale。

输入复刻 cases.py BENCH_CASES（e∈{4,8,32}, token_num_padded=256, hidden_dim=4096,
group_size=128, counts=[256]*e, 输入 dtype=bf16）。IS_INPLACE=True。
"""

import torch

OP_NAME = "silu_and_mul_masked_post_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(e, 256, 4096, 128) for e in (4, 8, 32)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_quant_fwd as _f,
        )
    except Exception:
        return None

    def _op(input, output, output_scale, quant_group_size, masked_m):
        _f(input, output, output_scale, quant_group_size, masked_m)
        return output, output_scale

    return _op


def grid():
    return [{"E": e, "T": t, "H": h, "group": grp} for (e, t, h, grp) in _SHAPES]


def build_inputs(binding, dtype, device):
    E, T, H, group = binding["E"], binding["T"], binding["H"], binding["group"]
    G = H // group
    g = torch.Generator(device=device).manual_seed(0)
    inp = torch.randn(E, T, H * 2, generator=g, device=device, dtype=torch.float32).to(dtype)
    masked_m = torch.tensor([T] * E, dtype=torch.int32, device=device)
    output = torch.empty(E, T, H, dtype=torch.float8_e4m3fn, device=device)
    output_scale = torch.empty(E, T, G, dtype=torch.float32, device=device)
    return (inp, output, output_scale, group, masked_m), {}


def key_shape(binding):
    return f"E{binding['E']}_T{binding['T']}_H{binding['H']}_g{binding['group']}"


def config(binding, dtype):
    E, T, H, group = binding["E"], binding["T"], binding["H"], binding["group"]
    G = H // group
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [E, T, H * 2], "dtype": dt},
            "output": {"shape": [E, T, H], "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "output_scale": {"shape": [E, T, G], "dtype": "torch.float32", "note": "原地写回"},
            "quant_group_size": {"scalar": group},
            "masked_m": {"shape": [E], "dtype": "torch.int32"},
        },
        "outputs": {
            "output": {"shape": [E, T, H], "dtype": "torch.float8_e4m3fn"},
            "output_scale": {"shape": [E, T, G], "dtype": "torch.float32"},
        },
        "dims": {"E": E, "T": T, "H": H, "H2": H * 2, "group": group, "G": G},
    }
