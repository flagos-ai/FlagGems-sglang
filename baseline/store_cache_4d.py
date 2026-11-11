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

"""Native SGLang v0.5.21 collection workload for store_cache_4d."""

import torch

OP_NAME = "store_cache_4d"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2, 3)

SHAPES = [(n, 8, 128) for n in (1, 8, 64, 512)]
MORE_SHAPES = [(n, 8, 128) for n in (4096,)]
_NUM_PAGES = 4096
_PAGE_SIZE = 64


def native():
    try:
        from sglang.kernels.ops.kvcache.kvcache import store_cache as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("n", "head_num", "head_dim"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    n, head_num, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    k_view = torch.zeros(
        _NUM_PAGES,
        _PAGE_SIZE,
        head_num,
        head_dim,
        dtype=cur_dtype,
        device=device,
    )
    v_view = torch.zeros_like(k_view)
    cache_k = torch.randn(
        n, head_num, head_dim, device=device, generator=g
    ).to(cur_dtype)
    cache_v = torch.randn(
        n, head_num, head_dim, device=device, generator=g
    ).to(cur_dtype)
    loc = torch.randperm(
        _NUM_PAGES * _PAGE_SIZE, device="cpu", generator=perm_g
    )[:n].to(device=device, dtype=torch.int64)
    return k_view, v_view, cache_k, cache_v, loc, _PAGE_SIZE


def build_inputs(binding, dtype, device):
    shape = (binding["n"], binding["head_num"], binding["head_dim"])
    k_cache, v_cache, k, v, loc, _ = build_case(shape, dtype, device)
    width = k.shape[-2] * k.shape[-1]
    return (
        k.view(-1, width),
        v.view(-1, width),
        k_cache.view(-1, width),
        v_cache.view(-1, width),
        loc,
    ), {"reserved_skip_index": -1}


def key_shape(binding):
    return [binding["n"], binding["head_num"], binding["head_dim"]]


def config(binding, dtype):
    n, head_num, head_dim = (
        binding["n"],
        binding["head_num"],
        binding["head_dim"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "k": tensor([n, head_num * head_dim]),
            "v": tensor([n, head_num * head_dim]),
            "k_cache": tensor([_NUM_PAGES * _PAGE_SIZE, head_num * head_dim]),
            "v_cache": tensor([_NUM_PAGES * _PAGE_SIZE, head_num * head_dim]),
            "indices": tensor([n], "torch.int64"),
            "reserved_skip_index": {"scalar": -1},
        },
        "outputs": {
            "k_cache": tensor([_NUM_PAGES * _PAGE_SIZE, head_num * head_dim]),
            "v_cache": tensor([_NUM_PAGES * _PAGE_SIZE, head_num * head_dim]),
        },
        "dims": dict(binding),
    }
