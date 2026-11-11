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

"""Native SGLang collection workload for fused_pack_qkv."""

import torch

OP_NAME = "fused_pack_qkv"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(1, s, 16, 128) for s in (8, 128, 1024)]
MORE_SHAPES = [(1, s, 16, 128) for s in (8192,)]
KEEP = 0.6


def native():
    """解析 fused_pack_qkv。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.varlen_pack_pad import (
            fused_pack_qkv as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton import (
                fused_pack_qkv as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "seq": seq, "num_heads": num_heads, "head_dim": head_dim}
        for (bs, seq, num_heads, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    bs, seq, num_heads, head_dim = (
        binding["bs"],
        binding["seq"],
        binding["num_heads"],
        binding["head_dim"],
    )
    g = torch.Generator(device=device).manual_seed(9)

    def r():
        return torch.randn(
            bs, seq, num_heads, head_dim, device=device, generator=g
        ).to(dtype)

    total = bs * seq
    n_valid = max(1, int(total * KEEP))
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    indices = (
        torch.randperm(
            total, generator=torch.Generator(device="cpu").manual_seed(9)
        )[:n_valid]
        .to(torch.int64)
        .to(device)
    )
    return (r(), r(), r(), indices), {}


def key_shape(binding):
    bs, seq = binding["bs"], binding["seq"]
    return f"bs{bs}_seq{seq}_h{binding['num_heads']}_d{binding['head_dim']}"


def config(binding, dtype):
    bs, seq = binding["bs"], binding["seq"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    n_valid = max(1, int(bs * seq * KEEP))
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [bs, seq, num_heads, head_dim], "dtype": dt},
            "k": {"shape": [bs, seq, num_heads, head_dim], "dtype": dt},
            "v": {"shape": [bs, seq, num_heads, head_dim], "dtype": dt},
            "indices": {
                "shape": [n_valid],
                "dtype": "torch.int64",
                "note": "n_valid = max(1, int(bs*seq*0.6))，数据相关",
            },
        },
        "outputs": {
            "q_packed": {"shape": [n_valid, num_heads, head_dim], "dtype": dt},
            "k_packed": {"shape": [n_valid, num_heads, head_dim], "dtype": dt},
            "v_packed": {"shape": [n_valid, num_heads, head_dim], "dtype": dt},
        },
        "dims": {
            "bs": bs,
            "seq": seq,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "keep": KEEP,
            "n_valid": n_valid,
        },
    }
