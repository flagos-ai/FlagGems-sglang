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

"""Native SGLang collection workload for get_mla_kv_buffer."""

import torch

OP_NAME = "get_mla_kv_buffer"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2, 3)

SHAPES = [(n, 512, 64) for n in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(n, 512, 64) for n in (16384,)]
_NUM_SLOTS = 65536


def native():
    """解析 get_mla_kv_buffer_triton。"""
    try:
        from sglang.kernels.ops.kvcache.mla_buffer import (
            get_mla_kv_buffer_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "n_loc": n_loc,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
            **{"num_slots": _NUM_SLOTS},
        }
        for (n_loc, nope_dim, rope_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    n_loc, nope_dim, rope_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    kv_buffer = torch.randn(
        _NUM_SLOTS, nope_dim + rope_dim, device=device, generator=g
    ).to(cur_dtype)
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    loc = (
        torch.randperm(
            _NUM_SLOTS, generator=torch.Generator(device="cpu").manual_seed(9)
        )[:n_loc]
        .to(torch.int64)
        .to(device)
    )
    nope = torch.zeros(n_loc, nope_dim, dtype=cur_dtype, device=device)
    rope = torch.zeros(n_loc, rope_dim, dtype=cur_dtype, device=device)
    return kv_buffer, loc, nope, rope


def build_inputs(binding, dtype, device):
    shape = (binding["n_loc"], binding["nope_dim"], binding["rope_dim"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"nloc{binding['n_loc']}_slots{binding['num_slots']}"
        f"_nope{binding['nope_dim']}_rope{binding['rope_dim']}"
    )


def config(binding, dtype):
    n_loc, num_slots = binding["n_loc"], binding["num_slots"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "kv_buffer": {
                "shape": [num_slots, nope_dim + rope_dim],
                "dtype": dt,
            },
            "loc": {"shape": [n_loc], "dtype": "torch.int64"},
            "cache_k_nope": {
                "shape": [n_loc, nope_dim],
                "dtype": dt,
                "note": "原地写回",
            },
            "cache_k_rope": {
                "shape": [n_loc, rope_dim],
                "dtype": dt,
                "note": "原地写回",
            },
        },
        "outputs": {
            "cache_k_nope": {"shape": [n_loc, nope_dim], "dtype": dt},
            "cache_k_rope": {"shape": [n_loc, rope_dim], "dtype": dt},
        },
        "dims": {
            "n_loc": n_loc,
            "num_slots": num_slots,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        },
    }
