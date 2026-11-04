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

"""gemma_qkv_rmsnorm baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.layernorm.gemma4_fused_ops.gemma_qkv_rmsnorm
    调用形态复刻 problems/activation_norm/gemma_qkv_rmsnorm/baseline.py：
        q = q.clone(); k = k.clone(); v = v.clone()
        gemma_qkv_rmsnorm(q, k, v, q_weight, k_weight,
                          num_q_heads, num_kv_heads, head_dim, eps=eps)
    原地写回 q/k/v -> IS_INPLACE=True；NCU 采样前重建实参避免 warmup 污染。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：_case(m, 32, 8, 128)。
    q = randn(m, nq*head_dim); k/v = randn(m, nkv*head_dim)（非 split_view）
    q_weight/k_weight = randn(head_dim)。同一 generator 顺序取数。
    dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "gemma_qkv_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 原地写回 q/k/v

_EPS = 1e-6
# BENCH_CASES: _case(m, 32, 8, 128) for m in (1, 8, 64, 512, 4096)
_SHAPES = [(m, 32, 8, 128) for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 gemma4_fused_ops.gemma_qkv_rmsnorm。"""
    try:
        from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
            gemma_qkv_rmsnorm as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "nq": nq, "nkv": nkv, "head_dim": hd}
        for (m, nq, nkv, hd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, nq, nkv, head_dim = (
        binding["m"], binding["nq"], binding["nkv"], binding["head_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)

    def _t(cols):
        return torch.randn(
            m, cols, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    q = _t(nq * head_dim)
    k = _t(nkv * head_dim)
    v = _t(nkv * head_dim)
    q_weight = torch.randn(
        head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    k_weight = torch.randn(
        head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    # baseline 传 clone 后的 q/k/v；采集侧新建等价。
    return (q, k, v, q_weight, k_weight, nq, nkv, head_dim), {"eps": _EPS}


def key_shape(binding):
    return "m{m}_nq{nq}_nkv{nkv}_hd{head_dim}".format(**binding)


def config(binding, dtype):
    m, nq, nkv, head_dim = (
        binding["m"], binding["nq"], binding["nkv"], binding["head_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [m, nq * head_dim], "dtype": dt, "note": "原地写回"},
            "k": {"shape": [m, nkv * head_dim], "dtype": dt, "note": "原地写回"},
            "v": {"shape": [m, nkv * head_dim], "dtype": dt, "note": "原地写回"},
            "q_weight": {"shape": [head_dim], "dtype": dt},
            "k_weight": {"shape": [head_dim], "dtype": dt},
            "num_q_heads": {"scalar": nq},
            "num_kv_heads": {"scalar": nkv},
            "head_dim": {"scalar": head_dim},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "q": {"shape": [m, nq * head_dim], "dtype": dt},
            "k": {"shape": [m, nkv * head_dim], "dtype": dt},
            "v": {"shape": [m, nkv * head_dim], "dtype": dt},
        },
        "dims": {"m": m, "num_q_heads": nq, "num_kv_heads": nkv, "head_dim": head_dim},
    }
