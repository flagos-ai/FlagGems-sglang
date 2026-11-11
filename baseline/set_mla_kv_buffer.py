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

"""Native SGLang collection workload for set_mla_kv_buffer."""

import torch

OP_NAME = "set_mla_kv_buffer"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(n, 512, 64) for n in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(n, 512, 64) for n in (16384,)]
_NUM_SLOTS = 65536


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        import triton
        from sglang.kernels.ops.kvcache.mla_buffer import (
            set_mla_kv_buffer_kernel as _kernel,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _call(kv_buffer, loc, cache_k_nope, cache_k_rope):
        nope_dim = cache_k_nope.shape[-1]
        rope_dim = cache_k_rope.shape[-1]
        block = triton.next_power_of_2(nope_dim + rope_dim)
        _kernel[(loc.numel(), 1)](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK=block,
            DCP_RANK=0,
            DCP_WORLD_SIZE=1,
        )
        return kv_buffer

    return _call


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
    kv_buffer = torch.zeros(
        _NUM_SLOTS, nope_dim + rope_dim, dtype=cur_dtype, device=device
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    loc = (
        torch.randperm(
            _NUM_SLOTS, generator=torch.Generator(device="cpu").manual_seed(9)
        )[:n_loc]
        .to(torch.int64)
        .to(device)
    )
    nope = torch.randn(n_loc, nope_dim, device=device, generator=g).to(
        cur_dtype
    )
    rope = torch.randn(n_loc, rope_dim, device=device, generator=g).to(
        cur_dtype
    )
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
                "note": "原地写回",
            },
            "loc": {"shape": [n_loc], "dtype": "torch.int64"},
            "cache_k_nope": {"shape": [n_loc, nope_dim], "dtype": dt},
            "cache_k_rope": {"shape": [n_loc, rope_dim], "dtype": dt},
        },
        "outputs": {
            "kv_buffer": {
                "shape": [num_slots, nope_dim + rope_dim],
                "dtype": dt,
            },
        },
        "dims": {
            "n_loc": n_loc,
            "num_slots": num_slots,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        },
    }
