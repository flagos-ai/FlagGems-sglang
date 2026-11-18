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

"""Native SGLang collection workload for hadamard_transform."""

import torch

OP_NAME = "hadamard_transform"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SCALE = 1.0
_SEED = 9
SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (128, 512, 2048)]
MORE_SHAPES = [(4096, n) for n in (128, 512, 2048)]


def native():
    """解析 hadamard_transform。"""
    try:
        from sglang.kernels.ops.quantization.hadamard import (
            hadamard_transform as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "n": n} for (m, n) in dict.fromkeys((*SHAPES, *MORE_SHAPES))
    ]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(m, n, device=device, generator=g).to(dtype)
    return (x, _SCALE), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "scale": {"scalar": _SCALE},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
