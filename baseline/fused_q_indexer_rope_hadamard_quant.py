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

"""fused_q_indexer_rope_hadamard_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsv4.elementwise.
        fused_q_indexer_rope_hadamard_quant
    调用形态复刻 problems/quantization/fused_q_indexer_rope_hadamard_quant/baseline.py：
        fused_q_indexer_rope_hadamard_quant(q_input, weight, weight_scale,
                                            freqs_cis, positions)
    返回 (q_fp8, w) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（h=16）：
    q_input=randn(b,h,128).to(bf16)；weight=rand(b,h).to(bf16)；weight_scale=1.0；
    freqs_cis=polar(1, rand(4096,32)*2pi) complex64；positions=randint(0,4096,(b,)) int32。
BENCH: _case(b,16) for b in (1,8,64,512,4096)。
"""

import torch

OP_NAME = "fused_q_indexer_rope_hadamard_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_H = 16
_MAX_SEQ = 4096
_WEIGHT_SCALE = 1.0
_SHAPES = [{"b": b, "h": _H} for b in (1, 8, 64, 512, 4096)]


def native():
    """解析 fused_q_indexer_rope_hadamard_quant。"""
    try:
        from sglang.kernels.ops.attention.dsv4.elementwise import (
            fused_q_indexer_rope_hadamard_quant as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    b, h = binding["b"], binding["h"]
    g = torch.Generator(device=device).manual_seed(0)
    q_input = torch.randn(
        b, h, 128, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    weight = torch.rand(b, h, dtype=torch.float32, device=device, generator=g).to(dtype)
    angle = torch.rand(_MAX_SEQ, 32, device=device, generator=g) * 6.283185307
    freqs_cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
    positions = torch.randint(
        0, _MAX_SEQ, (b,), device=device, generator=g, dtype=torch.int32
    )
    return (
        q_input.contiguous(),
        weight.contiguous(),
        _WEIGHT_SCALE,
        freqs_cis,
        positions,
    ), {}


def key_shape(binding):
    return f"b{binding['b']}_h{binding['h']}"


def config(binding, dtype):
    b, h = binding["b"], binding["h"]
    dt = str(dtype)
    return {
        "inputs": {
            "q_input": {"shape": [b, h, 128], "dtype": dt},
            "weight": {"shape": [b, h], "dtype": dt},
            "weight_scale": {"scalar": _WEIGHT_SCALE},
            "freqs_cis": {"shape": [_MAX_SEQ, 32], "dtype": "torch.complex64"},
            "positions": {"shape": [b], "dtype": "torch.int32"},
        },
        "outputs": {
            "q_fp8": {"shape": [b, h, 128], "dtype": "torch.float8_e4m3fn"},
            "w": {"shape": [b, h], "dtype": "torch.float32"},
        },
        "dims": {"b": b, "h": h, "max_seq": _MAX_SEQ},
    }
