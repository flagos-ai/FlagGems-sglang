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

"""Native SGLang collection workload for qk_gemma_rmsnorm_rope."""

import torch

OP_NAME = "qk_gemma_rmsnorm_rope"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, 32, 8, 128, 128) for m in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 32, 8, 128, 128)]
_EPS = 1e-6
_MAX_SEQ = 4096


def native():
    """解析 attention.minimax_m3_qk_norm_rope.qk_gemma_rmsnorm_rope。"""
    try:
        from sglang.kernels.ops.attention.minimax_m3_qk_norm_rope import (
            qk_gemma_rmsnorm_rope as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "m": m,
            "nq": nq,
            "nkv": nkv,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        }
        for (m, nq, nkv, head_dim, rotary_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    m, nq, nkv, head_dim, rotary_dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    angle = (
        torch.rand(_MAX_SEQ, rotary_dim // 2, device=device, generator=g)
        * 6.28
    )
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(cur_dtype)
    positions = torch.randint(
        0, _MAX_SEQ, (m,), device=device, generator=g, dtype=torch.int32
    )
    # is_neox_style is a bool, which unpack_to_args_kwargs only forwards as
    # part of a kwargs dict.
    return (
        _t(m, nq * head_dim),
        _t(m, nkv * head_dim),
        (0.5 * _t(head_dim).float()).to(cur_dtype),
        (0.5 * _t(head_dim).float()).to(cur_dtype),
        positions,
        cos_sin_cache,
        _EPS,
        head_dim,
        rotary_dim,
        {"is_neox_style": True},
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["m"],
        binding["nq"],
        binding["nkv"],
        binding["head_dim"],
        binding["rotary_dim"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return (*args, kwargs["is_neox_style"]), {}


def key_shape(binding):
    return "m{m}_nq{nq}_nkv{nkv}_hd{head_dim}_rd{rotary_dim}".format(**binding)


def config(binding, dtype):
    m, nq, nkv, head_dim, rotary_dim = (
        binding["m"],
        binding["nq"],
        binding["nkv"],
        binding["head_dim"],
        binding["rotary_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [m, nq * head_dim], "dtype": dt},
            "k": {"shape": [m, nkv * head_dim], "dtype": dt},
            "q_weight": {"shape": [head_dim], "dtype": dt},
            "k_weight": {"shape": [head_dim], "dtype": dt},
            "positions": {"shape": [m], "dtype": "torch.int32"},
            "cos_sin_cache": {"shape": [_MAX_SEQ, rotary_dim], "dtype": dt},
            "eps": {"scalar": _EPS},
            "head_dim": {"scalar": head_dim},
            "rotary_dim": {"scalar": rotary_dim},
            "is_neox_style": {"scalar": True},
        },
        "outputs": {
            "q_out": {"shape": [m, nq * head_dim], "dtype": dt},
            "k_out": {"shape": [m, nkv * head_dim], "dtype": dt},
        },
        "dims": {
            "m": m,
            "num_q_heads": nq,
            "num_kv_heads": nkv,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "is_neox_style": True,
        },
    }
