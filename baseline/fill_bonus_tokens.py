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

"""Native SGLang collection workload for fill_bonus_tokens."""

import torch

OP_NAME = "fill_bonus_tokens"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2,)

SHAPES = [(batch, 8) for batch in (1, 8, 64, 512)]
MORE_SHAPES = [(batch, 8) for batch in (4096,)]


def native():
    """解析 fill_bonus_tokens_func。"""
    try:
        from sglang.kernels.ops.speculative.eagle import (
            fill_bonus_tokens_func as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": batch, "width": width}
        for (batch, width) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, width = shape
    g = torch.Generator(device=device).manual_seed(9)
    tokens = torch.randint(
        0,
        128000,
        (batch, width),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    lens = torch.randint(
        1,
        width + 1,
        (batch,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    bonus = torch.zeros(batch, dtype=torch.int64, device=device)
    return tokens, lens, bonus, width


def build_inputs(binding, dtype, device):
    shape = (binding["batch"], binding["width"])
    tokens, lens, bonus, width = build_case(shape, dtype, device)
    return (tokens, lens, bonus, width, lens.shape[0]), {}


def key_shape(binding):
    return [binding["batch"], binding["width"]]


def config(binding, dtype):
    batch, width = binding["batch"], binding["width"]
    return {
        "inputs": {
            "accept_tokens": {"shape": [batch, width], "dtype": "torch.int64"},
            "accept_lens": {"shape": [batch], "dtype": "torch.int64"},
            "bonus_tokens": {
                "shape": [batch],
                "dtype": "torch.int64",
                "note": "原地写回",
            },
            "accept_stride": {"scalar": width},
            "batch": {"scalar": batch},
        },
        "outputs": {
            "bonus_tokens": {"shape": [batch], "dtype": "torch.int64"},
        },
        "dims": {"batch": batch, "width": width},
    }
