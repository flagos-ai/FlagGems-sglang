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

"""fused_norm_rope_inplace baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.attention.deepseek_v4_rope.fused_norm_rope_inplace_triton
    调用形态复刻 problems/rope/fused_norm_rope_inplace/baseline.py：
        kv = kv.clone()
        fused_norm_rope_inplace_triton(kv, weight, eps, freqs_cis, positions=positions)
    原地写回 kv 并返回 -> IS_INPLACE=True；NCU 采样前重建实参避免 warmup 污染。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：_case(m, 576, 64)。
    kv = randn(m, head_dim).contiguous().to(dtype)
    weight = (0.5 * randn(head_dim)).to(dtype)
    freqs_cis = polar(ones, rand(m, rope_dim//2)*2π).to(complex64)（positions=None）
    eps=1e-6。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_norm_rope_inplace"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 原地写回 kv

_EPS = 1e-6
_2PI = 6.283185307
# BENCH_CASES: _case(m, 576, 64) for m in (1, 8, 64, 512, 4096, 16384)
_SHAPES = [(m, 576, 64) for m in (1, 8, 64, 512, 4096, 16384)]


def native():
    """解析 attention.deepseek_v4_rope.fused_norm_rope_inplace_triton。"""
    try:
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            fused_norm_rope_inplace_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "head_dim": hd, "rope_dim": rd} for (m, hd, rd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, head_dim, rope_dim = (
        binding["m"], binding["head_dim"], binding["rope_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    kv = torch.randn(
        m, head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    weight = (
        0.5 * torch.randn(head_dim, dtype=torch.float32, device=device, generator=g)
    ).to(dtype)
    angle = torch.rand(m, rope_dim // 2, device=device, generator=g) * _2PI
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    # baseline 传 clone 后的 kv；采集侧新建等价。positions=None。
    return (kv.contiguous(), weight, _EPS, freqs_cis), {"positions": None}


def key_shape(binding):
    return "m{m}_hd{head_dim}_rd{rope_dim}".format(**binding)


def config(binding, dtype):
    m, head_dim, rope_dim = (
        binding["m"], binding["head_dim"], binding["rope_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "kv": {"shape": [m, head_dim], "dtype": dt, "note": "原地写回"},
            "weight": {"shape": [head_dim], "dtype": dt},
            "eps": {"scalar": _EPS},
            "freqs_cis": {"shape": [m, rope_dim // 2], "dtype": "torch.complex64"},
            "positions": {"scalar": None},
        },
        "outputs": {
            "kv": {"shape": [m, head_dim], "dtype": dt},
        },
        "dims": {"m": m, "head_dim": head_dim, "rope_dim": rope_dim},
    }
