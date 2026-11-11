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

"""Native SGLang collection workload for per_group_transpose."""

import torch

OP_NAME = "per_group_transpose"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

NUM_EXPERTS = 8
SHAPES = [(k, n) for k in (128, 512, 4096) for n in (16, 128, 1024)]


def native():
    """解析 per_group_transpose。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_group_transpose as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"k": k, "n": rows_per_expert} for (k, rows_per_expert) in SHAPES]


def build_case(shape, cur_dtype, device):
    k, rows_per_expert = shape
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(
        rows_per_expert * NUM_EXPERTS,
        k,
        generator=g,
        device=device,
        dtype=cur_dtype,
    ).contiguous()
    expert_offsets = torch.tensor(
        [i * rows_per_expert for i in range(NUM_EXPERTS + 1)],
        dtype=torch.int32,
        device=device,
    )
    return a, expert_offsets


def build_inputs(binding, dtype, device):
    shape = (binding["k"], binding["n"])
    return (*build_case(shape, dtype, device), 1), {}


def key_shape(binding):
    n = binding["n"]
    return [8 * n, binding["k"]]


def config(binding, dtype):
    k, n = binding["k"], binding["n"]
    m = 8 * n
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [m, k], "dtype": dt},
            "expert_offsets": {
                "shape": [9],
                "dtype": "torch.int32",
                "note": "cumsum([0]+[n]*8)，num_groups+1",
            },
            "m_alignment": {"scalar": 1},
        },
        "outputs": {
            "out": {"shape": [m, k], "dtype": dt, "note": "逐组转置结果"},
        },
        "dims": {"m": m, "k": k, "n_per_group": n, "num_groups": 8},
    }
