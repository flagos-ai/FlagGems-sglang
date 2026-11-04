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

"""fused_qk_gemma_rmsnorm_rope_gate baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate.fused_qk_gemma_rmsnorm_rope_gate
    调用形态复刻 problems/rope/fused_qk_gemma_rmsnorm_rope_gate/baseline.py：
        fused_qk_gemma_rmsnorm_rope_gate(
            q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps,
            num_q_heads, num_kv_heads, head_dim, rotary_dim, has_gate=has_gate)
    返回张量元组，不写回入参 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：_case(t, 32, 8, 128, 128)（has_gate=True）。
    q_cols = nq*(2 if has_gate else 1)*head_dim
    q_gate = randn(t, q_cols).contiguous().to(dtype)
    k = randn(t, nkv*head_dim).contiguous().to(dtype)
    q_weight/k_weight = (0.5 * randn(head_dim)).to(dtype)
    cos_sin_cache = cat([cos(angle), sin(angle)], -1).to(dtype)，angle=rand(4096, rotary_dim//2)*6.28
    positions = randint(0, 4096, (t,), int32)，eps=1e-6。
    同一 generator 顺序取数。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_qk_gemma_rmsnorm_rope_gate"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_MAX_SEQ = 4096
# BENCH_CASES: _case(t, 32, 8, 128, 128) for t in (1, 8, 64, 512, 4096)
_SHAPES = [(t, 32, 8, 128, 128) for t in (1, 8, 64, 512, 4096)]
_HAS_GATE = True


def native():
    """解析 attention.fused_qk_rmsnorm_rope_gate.fused_qk_gemma_rmsnorm_rope_gate。"""
    try:
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"t": t, "nq": nq, "nkv": nkv, "head_dim": hd, "rotary_dim": rd}
        for (t, nq, nkv, hd, rd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t, nq, nkv, head_dim, rotary_dim = (
        binding["t"], binding["nq"], binding["nkv"],
        binding["head_dim"], binding["rotary_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    q_cols = nq * (2 if _HAS_GATE else 1) * head_dim
    angle = torch.rand(_MAX_SEQ, rotary_dim // 2, device=device, generator=g) * 6.28
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(dtype)
    q_gate = _t(t, q_cols).contiguous()
    k = _t(t, nkv * head_dim).contiguous()
    q_weight = (0.5 * _t(head_dim).float()).to(dtype)
    k_weight = (0.5 * _t(head_dim).float()).to(dtype)
    positions = torch.randint(
        0, _MAX_SEQ, (t,), device=device, generator=g, dtype=torch.int32
    )
    return (
        q_gate, k, q_weight, k_weight, cos_sin_cache, positions, _EPS,
        nq, nkv, head_dim, rotary_dim,
    ), {"has_gate": _HAS_GATE}


def key_shape(binding):
    return "t{t}_nq{nq}_nkv{nkv}_hd{head_dim}_rd{rotary_dim}".format(**binding)


def config(binding, dtype):
    t, nq, nkv, head_dim, rotary_dim = (
        binding["t"], binding["nq"], binding["nkv"],
        binding["head_dim"], binding["rotary_dim"],
    )
    q_cols = nq * (2 if _HAS_GATE else 1) * head_dim
    dt = str(dtype)
    return {
        "inputs": {
            "q_gate": {"shape": [t, q_cols], "dtype": dt},
            "k": {"shape": [t, nkv * head_dim], "dtype": dt},
            "q_weight": {"shape": [head_dim], "dtype": dt},
            "k_weight": {"shape": [head_dim], "dtype": dt},
            "cos_sin_cache": {"shape": [_MAX_SEQ, rotary_dim], "dtype": dt},
            "positions": {"shape": [t], "dtype": "torch.int32"},
            "eps": {"scalar": _EPS},
            "num_q_heads": {"scalar": nq},
            "num_kv_heads": {"scalar": nkv},
            "head_dim": {"scalar": head_dim},
            "rotary_dim": {"scalar": rotary_dim},
            "has_gate": {"scalar": _HAS_GATE},
        },
        "outputs": {
            "q_out": {"shape": [t, nq * head_dim], "dtype": dt},
            "k_out": {"shape": [t, nkv * head_dim], "dtype": dt},
        },
        "dims": {
            "t": t, "num_q_heads": nq, "num_kv_heads": nkv,
            "head_dim": head_dim, "rotary_dim": rotary_dim, "has_gate": _HAS_GATE,
        },
    }
