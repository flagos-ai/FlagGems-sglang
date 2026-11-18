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

"""Native SGLang collection workload for apply_token_bitmask."""

import torch

OP_NAME = "apply_token_bitmask"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(b, 152064) for b in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(3, 100), (7, 1000), (1, 32000)]


def native():
    """解析 apply_token_bitmask_inplace_triton。"""
    try:
        from sglang.kernels.ops.grammar.bitmask_ops import (
            apply_token_bitmask_inplace_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"B": B, "V": V} for (B, V) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    B, V = shape
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(B, V, generator=g, device=device, dtype=cur_dtype)
    words = (V + 31) // 32
    bitmask = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (B, words),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    return logits, bitmask


def build_inputs(binding, dtype, device):
    shape = (binding["B"], binding["V"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["B"], binding["V"]]


def config(binding, dtype):
    B, V = binding["B"], binding["V"]
    words = (V + 31) // 32
    dt = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [B, V], "dtype": dt, "note": "原地写回"},
            "bitmask": {"shape": [B, words], "dtype": "torch.int32"},
        },
        "outputs": {
            "logits": {"shape": [B, V], "dtype": dt},
        },
        "dims": {"B": B, "V": V, "words": words},
    }
