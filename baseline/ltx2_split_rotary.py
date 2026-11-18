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

"""Native SGLang collection workload for ltx2_split_rotary."""

import torch

OP_NAME = "ltx2_split_rotary"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(1, t, 16, 128) for t in (1, 64, 1024)]
MORE_SHAPES = [(1, t, 16, 128) for t in (8192,)]


def native():
    """解析 apply_ltx2_split_rotary_emb。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.rope.ltx2_rotary_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.ltx2_rotary import (
            apply_ltx2_split_rotary_emb as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.rope.ltx2_rotary_triton import (
                apply_ltx2_split_rotary_emb as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        for (batch, seq_len, num_heads, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    batch, seq_len, num_heads, head_dim = (
        binding["batch"],
        binding["seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    )
    half = head_dim // 2
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(
        batch, seq_len, num_heads * head_dim, device=device, generator=g
    ).to(dtype)
    ang = torch.randn(
        batch, num_heads, seq_len, half, device=device, generator=g
    )
    return (
        x,
        torch.cos(ang).to(dtype),
        torch.sin(ang).to(dtype),
    ), {}


def key_shape(binding):
    return [
        binding["batch"],
        binding["seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    ]


def config(binding, dtype):
    batch, seq_len = binding["batch"], binding["seq_len"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    half = head_dim // 2
    dt = str(dtype)
    return {
        "inputs": {
            "x": {
                "shape": [batch, seq_len, num_heads * head_dim],
                "dtype": dt,
            },
            "cos": {"shape": [batch, num_heads, seq_len, half], "dtype": dt},
            "sin": {"shape": [batch, num_heads, seq_len, half], "dtype": dt},
        },
        "outputs": {
            "out": {
                "shape": [batch, seq_len, num_heads * head_dim],
                "dtype": dt,
            },
        },
        "dims": {
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
    }
