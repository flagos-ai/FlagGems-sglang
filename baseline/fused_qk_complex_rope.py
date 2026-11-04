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

"""fused_qk_complex_rope baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.attention.vision_rope.apply_fused_qk_complex_rope
    调用形态复刻 problems/rope/fused_qk_complex_rope/baseline.py：
        apply_fused_qk_complex_rope(q, k, freqs_cis)
    返回 (q_out, k_out) 两张量，不写回入参 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：_case(t, 16, 128)（非 packed）。
    q/k = randn(t, heads, head_dim).to(dtype)
    freqs_cis = polar(ones, rand(t, head_dim//2)*2π).to(complex64)。
    dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_qk_complex_rope"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_2PI = 6.283185307
# BENCH_CASES: _case(t, 16, 128) for t in (1, 64, 1024, 4096, 16384)
_SHAPES = [(t, 16, 128) for t in (1, 64, 1024, 4096, 16384)]


def native():
    """解析 attention.vision_rope.apply_fused_qk_complex_rope。"""
    try:
        from sglang.kernels.ops.attention.vision_rope import (
            apply_fused_qk_complex_rope as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"tokens": t, "heads": h, "head_dim": hd} for (t, h, hd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    tokens, heads, head_dim = (
        binding["tokens"], binding["heads"], binding["head_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        tokens, heads, head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    k = torch.randn(
        tokens, heads, head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    angle = torch.rand(tokens, head_dim // 2, device=device, generator=g) * _2PI
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    return (q, k, freqs_cis), {}


def key_shape(binding):
    return "t{tokens}_h{heads}_hd{head_dim}".format(**binding)


def config(binding, dtype):
    tokens, heads, head_dim = (
        binding["tokens"], binding["heads"], binding["head_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [tokens, heads, head_dim], "dtype": dt},
            "k": {"shape": [tokens, heads, head_dim], "dtype": dt},
            "freqs_cis": {"shape": [tokens, head_dim // 2], "dtype": "torch.complex64"},
        },
        "outputs": {
            "q_out": {"shape": [tokens, heads, head_dim], "dtype": dt},
            "k_out": {"shape": [tokens, heads, head_dim], "dtype": dt},
        },
        "dims": {"tokens": tokens, "heads": heads, "head_dim": head_dim},
    }
