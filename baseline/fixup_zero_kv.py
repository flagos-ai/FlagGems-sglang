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

"""Native SGLang collection workload for fixup_zero_kv."""

import torch

OP_NAME = "fixup_zero_kv"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)

SHAPES = [(bs, 16, 128, 4) for bs in (1, 8, 64, 512)]
MORE_SHAPES = [(bs, 16, 128, 4) for bs in (2048,)]
_ZERO_FRAC = 0.3


def native():
    """解析 sglang.kernels.ops.attention.fixup_zero_kv.fixup_zero_kv_rows。"""
    try:
        from sglang.kernels.ops.attention.fixup_zero_kv import (
            fixup_zero_kv_rows as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": bs,
            "num_heads": num_heads,
            "v_head_dim": v_head_dim,
            "max_q": max_q,
        }
        for (bs, num_heads, v_head_dim, max_q) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, num_heads, v_head_dim, max_q = shape
    g = torch.Generator(device=device).manual_seed(11)
    q_lens = torch.randint(
        1, max_q + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    cum = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cum[1:] = torch.cumsum(q_lens, dim=0)
    total = int(cum[-1])
    kv_lens = torch.randint(
        1, 4096, (bs,), dtype=torch.int32, device=device, generator=g
    )
    kv_lens[torch.rand(bs, device=device, generator=g) < _ZERO_FRAC] = 0
    out = torch.randn(
        total, num_heads, v_head_dim, device=device, generator=g
    ).to(cur_dtype)
    lse = torch.randn(total, num_heads, device=device, generator=g)
    return out, lse, kv_lens, cum, int(q_lens.max())


def build_inputs(binding, dtype, device):
    shape = (
        binding["bs"],
        binding["num_heads"],
        binding["v_head_dim"],
        binding["max_q"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return "bs%d_nh%d_vhd%d_mq%d" % (
        binding["bs"],
        binding["num_heads"],
        binding["v_head_dim"],
        binding["max_q"],
    )


def config(binding, dtype):
    bs = binding["bs"]
    num_heads, v_head_dim, max_q = (
        binding["num_heads"],
        binding["v_head_dim"],
        binding["max_q"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "out": {
                "shape": ["total_q", num_heads, v_head_dim],
                "dtype": dt,
                "note": "原地写回",
            },
            "lse": {
                "shape": ["total_q", num_heads],
                "dtype": "torch.float32",
                "note": "原地写回",
            },
            "kv_lens": {"shape": [bs], "dtype": "torch.int32"},
            "cum_seq_lens": {"shape": [bs + 1], "dtype": "torch.int32"},
            "max_seq_len": {"note": "标量 int = max(q_lens)，数据相关"},
        },
        "outputs": {
            "out": {"shape": ["total_q", num_heads, v_head_dim], "dtype": dt},
            "lse": {"shape": ["total_q", num_heads], "dtype": "torch.float32"},
        },
        "dims": {
            "bs": bs,
            "num_heads": num_heads,
            "v_head_dim": v_head_dim,
            "max_q": max_q,
            "note": "total_q = sum(randint(1,max_q+1)) 数据相关（seed=11）",
        },
    }
