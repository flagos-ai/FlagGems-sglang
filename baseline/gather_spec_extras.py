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

"""Native SGLang collection workload for gather_spec_extras."""

import torch

OP_NAME = "gather_spec_extras"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, 7168) for m in (1, 8, 64, 512)]
MORE_SHAPES = [(m, 7168) for m in (2048,)]
_POOL = 4096
_TOPK = 8


def native():
    """解析 sglang.kernels.ops.speculative.gather_spec_extras.gather_spec_extras。"""
    try:
        from sglang.kernels.ops.speculative.gather_spec_extras import (
            gather_spec_extras as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "hidden": hidden, **{"pool": _POOL, "topk": _TOPK}}
        for (m, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    m, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    indices = torch.randperm(_POOL, device="cpu", generator=perm_g)[:m].to(
        device=device, dtype=torch.int64
    )
    topk_p_buf = torch.rand(
        _POOL, _TOPK, dtype=torch.float32, device=device, generator=g
    )
    topk_index_buf = torch.randint(
        0,
        128000,
        (_POOL, _TOPK),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    output_tokens_buf = torch.randint(
        0,
        128000,
        (_POOL,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    hidden_states_buf = torch.randn(
        _POOL, hidden, device=device, generator=g
    ).to(cur_dtype)
    return (
        indices,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["hidden"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"m{binding['m']}_pool{binding['pool']}"
        f"_topk{binding['topk']}_h{binding['hidden']}"
    )


def config(binding, dtype):
    m, pool, topk, hidden = (
        binding["m"],
        binding["pool"],
        binding["topk"],
        binding["hidden"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "indices": {"shape": [m], "dtype": "torch.int64"},
            "topk_p_buf": {"shape": [pool, topk], "dtype": "torch.float32"},
            "topk_index_buf": {"shape": [pool, topk], "dtype": "torch.int64"},
            "output_tokens_buf": {"shape": [pool], "dtype": "torch.int64"},
            "hidden_states_buf": {"shape": [pool, hidden], "dtype": dt},
        },
        "outputs": {
            "topk_p": {"shape": [m, topk], "dtype": "torch.float32"},
            "topk_index": {"shape": [m, topk], "dtype": "torch.int64"},
            "output_tokens": {"shape": [m], "dtype": "torch.int64"},
            "hidden_states": {"shape": [m, hidden], "dtype": dt},
        },
        "dims": {"m": m, "pool": pool, "topk": topk, "hidden": hidden},
    }
