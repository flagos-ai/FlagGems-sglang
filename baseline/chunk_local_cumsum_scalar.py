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

"""Native SGLang collection workload for chunk_local_cumsum_scalar."""

import torch

OP_NAME = "chunk_local_cumsum_scalar"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(8, 16, 64, 32), (32, 4, 64, 64)]
MORE_SHAPES = [(2, 3, 16, 8), (3, 2, 32, 16)]


def native():
    """解析 sglang.kernels.ops.attention.fla.cumsum.chunk_local_cumsum_scalar。"""
    try:
        from sglang.kernels.ops.attention.fla.cumsum import (
            chunk_local_cumsum_scalar as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def binding_for_case(shape, reverse):
    result = dict(zip(("batch", "nchunks", "chunk_size", "nheads"), shape))
    result["reverse"] = reverse
    return result


def grid():
    return [
        binding_for_case(shape, reverse)
        for shape in (*SHAPES, *MORE_SHAPES)
        for reverse in (False, True)
    ]


def input_cases(shape, cur_dtype, device, reverse, scale=None):
    batch, nchunks, chunk_size, nheads = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        batch,
        nchunks * chunk_size,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    yield x, chunk_size, reverse, scale


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
    )
    ((x, chunk_size, reverse, scale),) = input_cases(
        shape, dtype, device, binding["reverse"]
    )
    return (x, chunk_size), {"reverse": reverse, "scale": scale}


def _forward_key(binding):
    t = binding["nchunks"] * binding["chunk_size"]
    return [binding["batch"], t, binding["nheads"]]


def key_shape(binding):
    key = _forward_key(binding)
    return f"{key}_reverse" if binding["reverse"] else key


def config(binding, dtype):
    b, nc, cs, h = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
    )
    t = nc * cs
    dt = str(dtype)
    return {
        "inputs": {
            "g": {"shape": [b, t, h], "dtype": dt},
            "chunk_size": {"scalar": cs},
            "reverse": {"scalar": binding["reverse"]},
            "scale": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [b, t, h], "dtype": dt},
        },
        "dims": {
            "batch": b,
            "nchunks": nc,
            "chunk_size": cs,
            "nheads": h,
            "t": t,
        },
    }
