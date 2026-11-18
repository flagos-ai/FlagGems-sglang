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

"""Native SGLang collection workload for silu_and_mul_masked."""

import torch

OP_NAME = "silu_and_mul_masked"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (1,)

SHAPES = [(e, 256, 4096, (256,) * e) for e in (4, 8, 32)]
MORE_SHAPES = [
    (4, 16, 32, (0, 3, 16, 9)),
    (8, 128, 256, (128, 0, 64, 1, 17, 128, 5, 90)),
]


def native():
    """解析 silu_and_mul_masked_fwd。"""
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "E": expert_num,
            "T": token_num_padded,
            "H": hidden_dim,
            "counts": counts,
        }
        for (expert_num, token_num_padded, hidden_dim, counts) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    expert_num, token_num_padded, hidden_dim, counts = shape
    g = torch.Generator(device=device).manual_seed(0)
    input = torch.randn(
        expert_num,
        token_num_padded,
        hidden_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    masked_m = torch.tensor(list(counts), dtype=torch.int32, device=device)
    return input, masked_m


def build_inputs(binding, dtype, device):
    shape = (binding["E"], binding["T"], binding["H"], binding["counts"])
    x, masked_m = build_case(shape, dtype, device)
    out = x.new_zeros((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.bfloat16)
    return (x, out, masked_m), {}


def key_shape(binding):
    return [binding["E"], binding["T"], binding["H"]] + list(binding["counts"])


def config(binding, dtype):
    E, T, H = binding["E"], binding["T"], binding["H"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [E, T, H], "dtype": dt},
            "output": {
                "shape": [E, T, H // 2],
                "dtype": "torch.bfloat16",
                "note": "原地写回",
            },
            "masked_m": {"shape": [E], "dtype": "torch.int32"},
        },
        "outputs": {
            "output": {"shape": [E, T, H // 2], "dtype": "torch.bfloat16"},
        },
        "dims": {"E": E, "T": T, "H": H},
    }
