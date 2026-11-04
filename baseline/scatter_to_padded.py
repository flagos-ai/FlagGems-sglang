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

"""scatter_to_padded baseline（方案 B，sglang）。

native：复刻 problems/diffusion/scatter_to_padded/baseline.py：
    sglang.kernels.ops.diffusion.triton.varlen_pack_pad.fused_scatter_to_padded
        fused_scatter_to_padded(out_unpad, inv_indices, batch_size, seqlen)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    每条序列保留一个随机前缀（lens = randint(1, seqlen+1, (batch,))），其余为 padding。
    total = sum(lens)；inv_indices[batch*seqlen] int32（前缀位置填 0..total-1，余 -1）；
    out_unpad = randn(total, heads, head_dim).bf16。
    BENCH 固定 seqlen=1024, heads=16, head_dim=128，遍历 batch∈{1,4,16,64}。
total 数据相关（随机 lens），config 里用符号描述。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "scatter_to_padded"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(b, 1024, 16, 128) for b in (1, 4, 16, 64)]


def native():
    """解析 fused_scatter_to_padded。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.varlen_pack_pad import (
            fused_scatter_to_padded as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton import (
                fused_scatter_to_padded as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "seqlen": s, "heads": h, "head_dim": d}
        for (b, s, h, d) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch, seqlen = binding["batch"], binding["seqlen"]
    heads, head_dim = binding["heads"], binding["head_dim"]
    g = torch.Generator(device=device).manual_seed(0)
    lens = torch.randint(
        1, seqlen + 1, (batch,), device=device, generator=g, dtype=torch.int64
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
        total, heads, head_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    return (out_unpad.contiguous(), inv, batch, seqlen), {}


def key_shape(binding):
    b, s = binding["batch"], binding["seqlen"]
    return f"b{b}_sl{s}_h{binding['heads']}_d{binding['head_dim']}"


def config(binding, dtype):
    batch, seqlen = binding["batch"], binding["seqlen"]
    heads, head_dim = binding["heads"], binding["head_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "out_unpad": {"shape": ["total", heads, head_dim], "dtype": dt,
                          "note": "total = sum(随机 lens)，数据相关"},
            "inv_indices": {"shape": [batch * seqlen], "dtype": "torch.int32"},
            "batch_size": {"scalar": batch},
            "seqlen": {"scalar": seqlen},
        },
        "outputs": {
            "out": {"shape": [batch, seqlen, heads, head_dim], "dtype": dt},
        },
        "dims": {
            "batch": batch, "seqlen": seqlen, "heads": heads, "head_dim": head_dim,
            "note": "total token 数由随机前缀长度决定，数据相关",
        },
    }
