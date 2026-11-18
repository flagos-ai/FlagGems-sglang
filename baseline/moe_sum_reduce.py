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

"""Native SGLang collection workload for moe_sum_reduce."""

import torch

OP_NAME = "moe_sum_reduce"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (1,)

ROUTED_SCALING_FACTOR = 2.5
SHAPES = [(t, 8, 7168) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(7, 2, 128), (83, 4, 512), (3, 8, 2048)]


def native():
    """解析 moe_sum_reduce_triton。"""
    try:
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            moe_sum_reduce_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"T": T, "top_k": top_k, "H": H}
        for (T, top_k, H) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    T, top_k, H = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        T, top_k, H, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    return x, ROUTED_SCALING_FACTOR


def build_inputs(binding, dtype, device):
    shape = (binding["T"], binding["top_k"], binding["H"])
    x, scale = build_case(shape, dtype, device)
    out = x.new_zeros((x.shape[0], x.shape[-1]))
    return (x, out, scale), {}


def key_shape(binding):
    return [binding["T"], binding["top_k"], binding["H"]]


def config(binding, dtype):
    T, top_k, H = binding["T"], binding["top_k"], binding["H"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [T, top_k, H], "dtype": dt},
            "output": {"shape": [T, H], "dtype": dt, "note": "原地写回"},
            "routed_scaling_factor": {"scalar": ROUTED_SCALING_FACTOR},
        },
        "outputs": {
            "output": {"shape": [T, H], "dtype": dt},
        },
        "dims": {"T": T, "top_k": top_k, "H": H},
    }
