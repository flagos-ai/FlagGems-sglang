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

"""Native SGLang v0.5.21 collection workload for layernorm_modulate."""

import torch

OP_NAME = "layernorm_modulate"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(b, s, 2240) for b in (1, 2) for s in (77, 1024, 4096)]
MORE_SHAPES = [(8, s, 2240) for s in (77, 1024, 4096)]
_EPS = 1e-6


def native():
    try:
        from sglang.kernels.kda_kernels.layernorm_modulate_triton import (
            fused_layernorm_modulate_raw as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("batch", "seq", "hidden"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    batch, seq, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    return _t(batch, seq, hidden), _t(batch, hidden), _t(batch, hidden), _EPS


def build_inputs(binding, dtype, device):
    shape = (binding["batch"], binding["seq"], binding["hidden"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["batch"], binding["seq"], binding["hidden"]]


def config(binding, dtype):
    batch, seq, hidden = binding["batch"], binding["seq"], binding["hidden"]

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "x": tensor([batch, seq, hidden]),
            "scale": tensor([batch, hidden]),
            "shift": tensor([batch, hidden]),
            "eps": {"scalar": _EPS},
        },
        "outputs": {"out": tensor([batch, seq, hidden])},
        "dims": dict(binding),
    }
