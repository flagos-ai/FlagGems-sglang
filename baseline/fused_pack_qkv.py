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

"""fused_pack_qkv baseline（方案 B，sglang）。

native：复刻 problems/diffusion/fused_pack_qkv/baseline.py：
    sglang.kernels.ops.diffusion.triton.varlen_pack_pad.fused_pack_qkv
        fused_pack_qkv(q, k, v, indices)
    返回 (q_packed, k_packed, v_packed) 新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    q/k/v = randn(bs, seq, num_heads, head_dim).bf16;
    indices = randperm(bs*seq)[:n_valid] int64，n_valid=max(1,int(bs*seq*0.6));
    BENCH 固定 bs=1, num_heads=16, head_dim=128，遍历 seq∈{8,128,1024,8192}。
indices 数据相关（n_valid 随 seq 变化），config 里用符号 n_valid 描述。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_pack_qkv"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NUM_HEADS = 16
_HEAD_DIM = 128
_KEEP = 0.6
_SHAPES = [(1, s) for s in (8, 128, 1024, 8192)]


def native():
    """解析 fused_pack_qkv。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.varlen_pack_pad import (
            fused_pack_qkv as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.layout.varlen_pack_pad_triton import (
                fused_pack_qkv as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "seq": s, "num_heads": _NUM_HEADS, "head_dim": _HEAD_DIM}
        for (bs, s) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    bs, seq = binding["bs"], binding["seq"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    g = torch.Generator(device=device).manual_seed(9)

    def r():
        return torch.randn(bs, seq, num_heads, head_dim, device=device, generator=g).to(dtype)

    total = bs * seq
    n_valid = max(1, int(total * _KEEP))
    indices = torch.randperm(total, device=device, generator=g)[:n_valid].to(torch.int64)
    return (r(), r(), r(), indices), {}


def key_shape(binding):
    bs, seq = binding["bs"], binding["seq"]
    return f"bs{bs}_seq{seq}_h{binding['num_heads']}_d{binding['head_dim']}"


def config(binding, dtype):
    bs, seq = binding["bs"], binding["seq"]
    num_heads, head_dim = binding["num_heads"], binding["head_dim"]
    n_valid = max(1, int(bs * seq * _KEEP))
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [bs, seq, num_heads, head_dim], "dtype": dt},
            "k": {"shape": [bs, seq, num_heads, head_dim], "dtype": dt},
            "v": {"shape": [bs, seq, num_heads, head_dim], "dtype": dt},
            "indices": {"shape": [n_valid], "dtype": "torch.int64",
                        "note": "n_valid = max(1, int(bs*seq*0.6))，数据相关"},
        },
        "outputs": {
            "q_packed": {"shape": [n_valid, num_heads, head_dim], "dtype": dt},
            "k_packed": {"shape": [n_valid, num_heads, head_dim], "dtype": dt},
            "v_packed": {"shape": [n_valid, num_heads, head_dim], "dtype": dt},
        },
        "dims": {
            "bs": bs, "seq": seq, "num_heads": num_heads, "head_dim": head_dim,
            "keep": _KEEP, "n_valid": n_valid,
        },
    }
