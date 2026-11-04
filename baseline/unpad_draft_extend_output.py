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

"""unpad_draft_extend_output baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.pad.unpad_draft_extend_output
    调用形态复刻 problems/attention/unpad_draft_extend_output/baseline.py：
        unpad_draft_extend_output(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q)
    返回新张量（去 padding），不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    seq_lens = randint(1, token_per_batch+1, (bs,), int32)，cu 前缀和
    raw_out = randn(bs, token_per_batch, num_heads, head_dim).bf16
    sum_seq_lens_q = int(cu[-1])；token_per_batch=8, num_heads=16, head_dim=128。seed=7。
dtype 取 cases.py 的 [bfloat16]。sum_seq_lens_q 数据相关。
"""

import torch

OP_NAME = "unpad_draft_extend_output"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_TOKEN_PER_BATCH = 8
_NUM_HEADS = 16
_HEAD_DIM = 128
_CASES = [1, 8, 64, 256, 1024]


def native():
    """解析 sglang.kernels.ops.attention.pad.unpad_draft_extend_output。"""
    try:
        from sglang.kernels.ops.attention.pad import unpad_draft_extend_output as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "token_per_batch": _TOKEN_PER_BATCH,
         "num_heads": _NUM_HEADS, "head_dim": _HEAD_DIM}
        for bs in _CASES
    ]


def build_inputs(binding, dtype, device):
    bs = binding["bs"]
    tpb, num_heads, head_dim = (
        binding["token_per_batch"], binding["num_heads"], binding["head_dim"],
    )
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(1, tpb + 1, (bs,), dtype=torch.int32, device=device, generator=g)
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    raw_out = torch.randn(
        bs, tpb, num_heads, head_dim, device=device, generator=g
    ).to(dtype)
    sum_seq_lens_q = int(cu[-1])
    return (raw_out, cu, seq_lens, sum_seq_lens_q), {}


def key_shape(binding):
    return "bs%d_tpb%d_nh%d_hd%d" % (
        binding["bs"], binding["token_per_batch"],
        binding["num_heads"], binding["head_dim"],
    )


def config(binding, dtype):
    bs = binding["bs"]
    tpb, num_heads, head_dim = (
        binding["token_per_batch"], binding["num_heads"], binding["head_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "raw_out": {"shape": [bs, tpb, num_heads, head_dim], "dtype": dt},
            "cu_seqlens_q": {"shape": [bs + 1], "dtype": "torch.int32"},
            "seq_lens_q": {"shape": [bs], "dtype": "torch.int32"},
            "sum_seq_lens_q": {"note": "标量 int = cu_seqlens_q[-1]，数据相关"},
        },
        "outputs": {
            "out": {"shape": ["sum_seq_lens_q", num_heads, head_dim], "dtype": dt,
                    "note": "去 padding 后长度数据相关"},
        },
        "dims": {
            "bs": bs, "token_per_batch": tpb,
            "num_heads": num_heads, "head_dim": head_dim,
            "note": "sum_seq_lens_q = sum(randint(1,tpb+1)) 数据相关（seed=7）",
        },
    }
