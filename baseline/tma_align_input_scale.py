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

"""Native SGLang collection workload for tma_align_input_scale."""

import torch

OP_NAME = "tma_align_input_scale"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, 56) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(m, 56) for m in (16384,)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            tma_align_input_scale as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "k_div_block": k_div_block}
        for (m, k_div_block) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    m, k_div_block = shape
    g = torch.Generator(device=device).manual_seed(9)
    # input_scale is an fp32 activation-scale tensor regardless of cur_dtype.
    input_scale = torch.rand(
        m, k_div_block, dtype=torch.float32, device=device, generator=g
    )
    return (input_scale,)


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["k_div_block"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["m"], binding["k_div_block"]]


def config(binding, dtype):
    m, k_div_block = binding["m"], binding["k_div_block"]
    return {
        "inputs": {
            "input_scale": {
                "shape": [m, k_div_block],
                "dtype": "torch.float32",
            },
        },
        "outputs": {
            "note": "TMA 对齐后的 input_scale（转置/padding 布局）",
        },
        "dims": {"m": m, "k_div_block": k_div_block},
    }
