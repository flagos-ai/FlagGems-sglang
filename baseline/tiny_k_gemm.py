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

"""Native SGLang v0.5.21 collection workload for tiny_k_gemm."""

import torch

OP_NAME = "tiny_k_gemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, n, 128) for m in (1, 2, 4, 8) for n in (1536, 3072)]
MORE_SHAPES = [(m, n, 128) for m in (16,) for n in (1536, 3072)]


def native():
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16 as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("m", "n", "k"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    m, n, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(m, k, device=device, generator=g).to(cur_dtype)
    w = torch.randn(n, k, device=device, generator=g).to(cur_dtype)
    # out_dtype is a torch.dtype, which unpack_to_args_kwargs forwards
    # positionally.
    return x, w, cur_dtype


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["n"], binding["k"])
    x, w, out_dtype = build_case(shape, dtype, device)
    return (x, w), {"out_dtype": out_dtype}


def key_shape(binding):
    return [binding["m"], binding["n"], binding["k"]]


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "x": tensor([m, k]),
            "w": tensor([n, k]),
            "out_dtype": {"scalar": str(dtype)},
        },
        "outputs": {"out": tensor([m, n])},
        "dims": dict(binding),
    }
