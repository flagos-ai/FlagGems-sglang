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

"""Native SGLang v0.5.21 collection workload for qk_head_layernorm."""

import torch

OP_NAME = "qk_head_layernorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(1, s, 32, 128) for s in (128, 1024, 4360)]
MORE_SHAPES = [(1, 16384, 32, 128)]
_EPS = 1e-6


def native():
    try:
        from sglang.kernels.kda_kernels.layernorm_modulate_triton import (
            fused_qk_head_layernorm as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("b", "s", "heads", "head_dim"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    b, s, heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t():
        return torch.randn(
            b,
            s,
            heads,
            head_dim,
            dtype=torch.float32,
            device=device,
            generator=g,
        ).to(cur_dtype)

    return _t(), _t(), _EPS


def build_inputs(binding, dtype, device):
    shape = (binding["b"], binding["s"], binding["heads"], binding["head_dim"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["b"], binding["s"], binding["heads"], binding["head_dim"]]


def config(binding, dtype):
    b, s, heads, head_dim = (
        binding["b"],
        binding["s"],
        binding["heads"],
        binding["head_dim"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "q": tensor([b, s, heads, head_dim]),
            "k": tensor([b, s, heads, head_dim]),
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "q": tensor([b, s, heads, head_dim]),
            "k": tensor([b, s, heads, head_dim]),
        },
        "dims": dict(binding),
    }
