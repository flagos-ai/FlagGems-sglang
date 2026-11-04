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

"""apply_rotary_emb baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.attention.deepseek_v4_rope.apply_rotary_emb_triton
    调用形态复刻 problems/rope/apply_rotary_emb/baseline.py：
        x = x.clone()
        apply_rotary_emb_triton(x, freqs_cis, positions=positions, inverse=inverse)
    原地写回 x 并返回 -> IS_INPLACE=True；NCU 采样前重建实参避免 warmup 污染。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：_case(t, 16, 64)。
    x = randn(t, heads, rope_dim).contiguous().to(dtype)
    freqs_cis = polar(ones, rand(t, rope_dim//2)*2π).to(complex64)（positions=None）
    inverse=False。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "apply_rotary_emb"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 原地写回 x

_2PI = 6.283185307
# BENCH_CASES: _case(t, 16, 64) for t in (1, 8, 64, 512, 4096, 16384)
_SHAPES = [(t, 16, 64) for t in (1, 8, 64, 512, 4096, 16384)]


def native():
    """解析 attention.deepseek_v4_rope.apply_rotary_emb_triton。"""
    try:
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            apply_rotary_emb_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"tokens": t, "heads": h, "rope_dim": rd} for (t, h, rd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    tokens, heads, rope_dim = (
        binding["tokens"], binding["heads"], binding["rope_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        tokens, heads, rope_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    angle = torch.rand(tokens, rope_dim // 2, device=device, generator=g) * _2PI
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    # baseline 传 clone 后的 x；采集侧新建等价。positions=None, inverse=False。
    return (x.contiguous(), freqs_cis), {"positions": None, "inverse": False}


def key_shape(binding):
    return "t{tokens}_h{heads}_rd{rope_dim}".format(**binding)


def config(binding, dtype):
    tokens, heads, rope_dim = (
        binding["tokens"], binding["heads"], binding["rope_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [tokens, heads, rope_dim], "dtype": dt, "note": "原地写回"},
            "freqs_cis": {"shape": [tokens, rope_dim // 2], "dtype": "torch.complex64"},
            "positions": {"scalar": None},
            "inverse": {"scalar": False},
        },
        "outputs": {
            "x": {"shape": [tokens, heads, rope_dim], "dtype": dt},
        },
        "dims": {"tokens": tokens, "heads": heads, "rope_dim": rope_dim},
    }
