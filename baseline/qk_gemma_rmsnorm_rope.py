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

"""qk_gemma_rmsnorm_rope baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.attention.minimax_m3_qk_norm_rope.qk_gemma_rmsnorm_rope
    调用形态复刻 problems/rope/qk_gemma_rmsnorm_rope/baseline.py：
        qk_gemma_rmsnorm_rope(
            q, k, q_weight, k_weight, positions, cos_sin_cache, eps,
            head_dim, rotary_dim, is_neox_style)
    返回张量元组，不写回入参 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：_case(m, 32, 8, 128, 128)（neox=True）。
    q = randn(m, nq*head_dim).contiguous().to(dtype)
    k = randn(m, nkv*head_dim).contiguous().to(dtype)
    q_weight/k_weight = (0.5 * randn(head_dim)).to(dtype)
    cos_sin_cache = cat([cos(angle), sin(angle)], -1).to(dtype)，angle=rand(4096, rotary_dim//2)*6.28
    positions = randint(0, 4096, (m,), int32)，eps=1e-6。
    同一 generator 顺序取数。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "qk_gemma_rmsnorm_rope"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_MAX_SEQ = 4096
# BENCH_CASES: _case(m, 32, 8, 128, 128) for m in (1, 8, 64, 512, 4096)
_SHAPES = [(m, 32, 8, 128, 128) for m in (1, 8, 64, 512, 4096)]
_NEOX = True


def native():
    """解析 attention.minimax_m3_qk_norm_rope.qk_gemma_rmsnorm_rope。"""
    try:
        from sglang.kernels.ops.attention.minimax_m3_qk_norm_rope import (
            qk_gemma_rmsnorm_rope as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "nq": nq, "nkv": nkv, "head_dim": hd, "rotary_dim": rd}
        for (m, nq, nkv, hd, rd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, nq, nkv, head_dim, rotary_dim = (
        binding["m"], binding["nq"], binding["nkv"],
        binding["head_dim"], binding["rotary_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    angle = torch.rand(_MAX_SEQ, rotary_dim // 2, device=device, generator=g) * 6.28
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(dtype)
    q = _t(m, nq * head_dim).contiguous()
    k = _t(m, nkv * head_dim).contiguous()
    q_weight = (0.5 * _t(head_dim).float()).to(dtype)
    k_weight = (0.5 * _t(head_dim).float()).to(dtype)
    positions = torch.randint(
        0, _MAX_SEQ, (m,), device=device, generator=g, dtype=torch.int32
    )
    return (
        q, k, q_weight, k_weight, positions, cos_sin_cache, _EPS,
        head_dim, rotary_dim, _NEOX,
    ), {}


def key_shape(binding):
    return "m{m}_nq{nq}_nkv{nkv}_hd{head_dim}_rd{rotary_dim}".format(**binding)


def config(binding, dtype):
    m, nq, nkv, head_dim, rotary_dim = (
        binding["m"], binding["nq"], binding["nkv"],
        binding["head_dim"], binding["rotary_dim"],
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
            "is_neox_style": {"scalar": _NEOX},
        },
        "outputs": {
            "q_out": {"shape": [m, nq * head_dim], "dtype": dt},
            "k_out": {"shape": [m, nkv * head_dim], "dtype": dt},
        },
        "dims": {
            "m": m, "num_q_heads": nq, "num_kv_heads": nkv,
            "head_dim": head_dim, "rotary_dim": rotary_dim, "is_neox_style": _NEOX,
        },
    }
