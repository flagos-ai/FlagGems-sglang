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

"""Native SGLang collection workload for fill_accept_out_cache_loc."""

import torch

OP_NAME = "fill_accept_out_cache_loc"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2,)

SHAPES = [(s,) for s in (8, 64, 512)]
MORE_SHAPES = [(4096,)]


def native():
    """解析 fill_accept_out_cache_loc_func。"""
    try:
        from sglang.kernels.ops.speculative.eagle import (
            fill_accept_out_cache_loc_func as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"size": size} for (size,) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    (size,) = shape
    g = torch.Generator(device=device).manual_seed(9)
    accept_index = torch.arange(size, dtype=torch.int64, device=device)
    accept_index[torch.rand(size, device=device, generator=g) > 0.5] = -1
    out_cache_loc = torch.randint(
        1,
        1 << 20,
        (size,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    return accept_index, out_cache_loc, accept_out_cache_loc, size


def build_inputs(binding, dtype, device):
    shape = (binding["size"],)
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["size"]]


def config(binding, dtype):
    size = binding["size"]
    return {
        "inputs": {
            "accept_index": {"shape": [size], "dtype": "torch.int64"},
            "out_cache_loc": {"shape": [size], "dtype": "torch.int64"},
            "accept_out_cache_loc": {
                "shape": [size],
                "dtype": "torch.int64",
                "note": "原地写回",
            },
            "size": {"scalar": size},
        },
        "outputs": {
            "accept_out_cache_loc": {"shape": [size], "dtype": "torch.int64"},
        },
        "dims": {"size": size},
    }
