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

"""ltx2_split_rotary baseline（方案 B，sglang）。

native：复刻 problems/diffusion/ltx2_split_rotary/baseline.py：
    sglang.kernels.ops.diffusion.triton.ltx2_rotary.apply_ltx2_split_rotary_emb
        apply_ltx2_split_rotary_emb(x, cos, sin)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(batch, seq_len, num_heads*head_dim).bf16;
    ang = randn(batch, num_heads, seq_len, head_dim//2); cos/sin = cos/sin(ang).bf16;
    BENCH 固定 batch=1, num_heads=16, head_dim=128，遍历 seq_len∈{1,64,1024,8192}。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "ltx2_split_rotary"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(1, t, 16, 128) for t in (1, 64, 1024, 8192)]


def native():
    """解析 apply_ltx2_split_rotary_emb。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.rope.ltx2_rotary_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.ltx2_rotary import (
            apply_ltx2_split_rotary_emb as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.rope.ltx2_rotary_triton import (
                apply_ltx2_split_rotary_emb as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "seq_len": t, "num_heads": h, "head_dim": d}
        for (b, t, h, d) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch, seq_len = binding["batch"], binding["seq_len"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    half = head_dim // 2
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(
        batch, seq_len, num_heads * head_dim, device=device, generator=g
    ).to(dtype)
    ang = torch.randn(batch, num_heads, seq_len, half, device=device, generator=g)
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    return (x, cos, sin), {}


def key_shape(binding):
    return [
        binding["batch"], binding["seq_len"],
        binding["num_heads"], binding["head_dim"],
    ]


def config(binding, dtype):
    batch, seq_len = binding["batch"], binding["seq_len"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    half = head_dim // 2
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [batch, seq_len, num_heads * head_dim], "dtype": dt},
            "cos": {"shape": [batch, num_heads, seq_len, half], "dtype": dt},
            "sin": {"shape": [batch, num_heads, seq_len, half], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [batch, seq_len, num_heads * head_dim], "dtype": dt},
        },
        "dims": {
            "batch": batch, "seq_len": seq_len,
            "num_heads": num_heads, "head_dim": head_dim,
        },
    }
