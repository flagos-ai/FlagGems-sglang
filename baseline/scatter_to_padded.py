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

"""Native SGLang collection workload for scatter_to_padded."""

import torch

OP_NAME = "scatter_to_padded"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(b, 1024, 16, 128) for b in (1, 4, 16)]
MORE_SHAPES = [(64, 1024, 16, 128)]


def native():
    """解析 fused_scatter_to_padded。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.varlen_pack_pad import (
            fused_scatter_to_padded as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton import (
                fused_scatter_to_padded as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "head_dim": head_dim,
        }
        for (batch, seqlen, heads, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, seqlen, heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    lens = torch.randint(
        1,
        seqlen + 1,
        (batch,),
        device=device,
        generator=g,
        dtype=torch.int64,
    )
    total = int(lens.sum())
    inv = torch.full((batch * seqlen,), -1, dtype=torch.int32, device=device)
    cursor = 0
    for b in range(batch):
        n = int(lens[b])
        inv[b * seqlen : b * seqlen + n] = torch.arange(
            cursor, cursor + n, dtype=torch.int32, device=device
        )
        cursor += n
    out_unpad = torch.randn(
        total,
        heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    return out_unpad, inv, batch, seqlen


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["seqlen"],
        binding["heads"],
        binding["head_dim"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    b, s = binding["batch"], binding["seqlen"]
    return f"b{b}_sl{s}_h{binding['heads']}_d{binding['head_dim']}"


def config(binding, dtype):
    batch, seqlen = binding["batch"], binding["seqlen"]
    heads, head_dim = binding["heads"], binding["head_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "out_unpad": {
                "shape": ["total", heads, head_dim],
                "dtype": dt,
                "note": "total = sum(随机 lens)，数据相关",
            },
            "inv_indices": {"shape": [batch * seqlen], "dtype": "torch.int32"},
            "batch_size": {"scalar": batch},
            "seqlen": {"scalar": seqlen},
        },
        "outputs": {
            "out": {"shape": [batch, seqlen, heads, head_dim], "dtype": dt},
        },
        "dims": {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "head_dim": head_dim,
            "note": "total token 数由随机前缀长度决定，数据相关",
        },
    }
