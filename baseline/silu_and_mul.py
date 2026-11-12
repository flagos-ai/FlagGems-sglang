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

"""Native SGLang collection workload for silu_and_mul."""

import torch

OP_NAME = "silu_and_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(bs, 4096) for bs in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(bs, d) for bs in (1, 8, 64, 512, 4096) for d in (1024, 8192)]


def native():
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            silu_and_mul_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "d": d} for (bs, d) in (*SHAPES, *MORE_SHAPES)]


def build_inputs(binding, dtype, device):
    bs, d = binding["bs"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        bs, 2 * d, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    return (hidden_states,), {}


def key_shape(binding):
    return [binding["bs"], 2 * binding["d"]]


def config(binding, dtype):
    bs, d = binding["bs"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "hidden_states": {"shape": [bs, 2 * d], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [bs, d], "dtype": dt},
        },
        "dims": {"bs": bs, "d": d, "hidden": 2 * d},
    }
