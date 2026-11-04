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

"""rotary_embedding baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.diffusion.triton.rotary.apply_rotary_embedding
    调用形态复刻 problems/diffusion/rotary_embedding/baseline.py：
        apply_rotary_embedding(x, cos, sin, interleaved)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/diffusion/rotary_embedding/cases.py 的 BENCH_CASES：
    x=randn(tokens, heads, head_size).bf16；ang=randn(tokens, head_size//2)；
    cos=cos(ang).bf16; sin=sin(ang).bf16; interleaved=False。seed=9。
dtype 取 benchmark/test_rotary_embedding.py 的 [bfloat16]。
"""

import torch

OP_NAME = "rotary_embedding"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: _case(t, seed=9) for t in (1,64,1024,8192,32768)，默认 heads=16, head_size=128
_SHAPES = [{"tokens": t, "heads": 16, "head_size": 128} for t in (1, 64, 1024, 8192, 32768)]
_SEED = 9


def native():
    """解析 apply_rotary_embedding。

    spec/baseline 坐标为 diffusion.triton.rotary；sglang 0.5.21 把该符号挪到了
    diffusion.rope.rotary_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.rotary import (
            apply_rotary_embedding as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.rope.rotary_triton import (
                apply_rotary_embedding as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    tokens, heads, head_size = binding["tokens"], binding["heads"], binding["head_size"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(tokens, heads, head_size, device=device, generator=g).to(dtype)
    ang = torch.randn(tokens, head_size // 2, device=device, generator=g)
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    return (x, cos, sin, False), {}


def key_shape(binding):
    return [binding["tokens"], binding["heads"], binding["head_size"]]


def config(binding, dtype):
    tokens, heads, head_size = binding["tokens"], binding["heads"], binding["head_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [tokens, heads, head_size], "dtype": dt},
            "cos": {"shape": [tokens, head_size // 2], "dtype": dt},
            "sin": {"shape": [tokens, head_size // 2], "dtype": dt},
            "interleaved": {"scalar": False},
        },
        "outputs": {
            "out": {"shape": [tokens, heads, head_size], "dtype": dt},
        },
        "dims": {"tokens": tokens, "heads": heads, "head_size": head_size},
    }
