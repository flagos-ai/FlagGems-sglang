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

"""Native SGLang collection workload for pad_draft_extend_query."""

import torch

OP_NAME = "pad_draft_extend_query"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (1,)

SHAPES = [(bs, 8, 16, 128) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 8, 16, 128) for bs in (1024,)]


def native():
    """解析 sglang.kernels.ops.attention.pad.pad_draft_extend_query。"""
    try:
        from sglang.kernels.ops.attention.pad import (
            pad_draft_extend_query as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": bs,
            "max_seq_len": max_seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        for (bs, max_seq_len, num_heads, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, max_seq_len, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1,
        max_seq_len + 1,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    total = int(cu[-1])
    q = torch.randn(total, num_heads, head_dim, device=device, generator=g).to(
        cur_dtype
    )
    padded_q = torch.zeros(
        bs, max_seq_len, num_heads, head_dim, dtype=cur_dtype, device=device
    )
    return q, padded_q, seq_lens, cu


def build_inputs(binding, dtype, device):
    shape = (
        binding["bs"],
        binding["max_seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return "bs%d_msl%d_nh%d_hd%d" % (
        binding["bs"],
        binding["max_seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    )


def config(binding, dtype):
    bs = binding["bs"]
    max_seq_len, num_heads, head_dim = (
        binding["max_seq_len"],
        binding["num_heads"],
        binding["head_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": ["total_q", num_heads, head_dim], "dtype": dt},
            "out": {
                "shape": [bs, max_seq_len, num_heads, head_dim],
                "dtype": dt,
                "note": "原地写回",
            },
            "seq_lens_q": {"shape": [bs], "dtype": "torch.int32"},
            "cu_seqlens_q": {"shape": [bs + 1], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {
                "shape": [bs, max_seq_len, num_heads, head_dim],
                "dtype": dt,
            },
        },
        "dims": {
            "bs": bs,
            "max_seq_len": max_seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "note": "total_q = sum(randint(1,max_seq_len+1)) 数据相关（seed=7）",
        },
    }
