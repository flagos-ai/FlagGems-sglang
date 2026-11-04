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

"""fixup_zero_kv baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.fixup_zero_kv.fixup_zero_kv_rows
    调用形态复刻 problems/attention/fixup_zero_kv/baseline.py：
        out_c, lse_c = out.clone(), lse.clone()
        fixup_zero_kv_rows(out_c, lse_c, kv_lens, cum_seq_lens, max_seq_len)
    kernel 原地修正 out_c / lse_c（故 baseline 先 clone）。IS_INPLACE=True。

输入构造复刻 problems/attention/fixup_zero_kv/cases.py 的 BENCH_CASES：
    q_lens = randint(1, max_q+1, (bs,), int32)，cum 前缀和；total = cum[-1]
    kv_lens = randint(1, 4096, (bs,), int32)，zero_frac=0.3 随机置 0
    out = randn(total, num_heads, v_head_dim).bf16；lse = randn(total, num_heads) fp32
    num_heads=16, v_head_dim=128, max_q=4。seed=11。
dtype 取 cases.py 的 [bfloat16]。total 数据相关。
"""

import torch

OP_NAME = "fixup_zero_kv"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地修正 out / lse

_NUM_HEADS = 16
_V_HEAD_DIM = 128
_MAX_Q = 4
_ZERO_FRAC = 0.3
_CASES = [1, 8, 64, 512, 2048]


def native():
    """解析 sglang.kernels.ops.attention.fixup_zero_kv.fixup_zero_kv_rows。"""
    try:
        from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "num_heads": _NUM_HEADS, "v_head_dim": _V_HEAD_DIM, "max_q": _MAX_Q}
        for bs in _CASES
    ]


def build_inputs(binding, dtype, device):
    bs = binding["bs"]
    num_heads, v_head_dim, max_q = (
        binding["num_heads"], binding["v_head_dim"], binding["max_q"],
    )
    g = torch.Generator(device=device).manual_seed(11)
    q_lens = torch.randint(1, max_q + 1, (bs,), dtype=torch.int32, device=device, generator=g)
    cum = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cum[1:] = torch.cumsum(q_lens, dim=0)
    total = int(cum[-1])
    kv_lens = torch.randint(1, 4096, (bs,), dtype=torch.int32, device=device, generator=g)
    mask = torch.rand(bs, device=device, generator=g) < _ZERO_FRAC
    kv_lens[mask] = 0
    out = torch.randn(total, num_heads, v_head_dim, device=device, generator=g).to(dtype)
    lse = torch.randn(total, num_heads, device=device, generator=g)
    max_seq_len = int(q_lens.max())
    return (out, lse, kv_lens, cum, max_seq_len), {}


def key_shape(binding):
    return "bs%d_nh%d_vhd%d_mq%d" % (
        binding["bs"], binding["num_heads"], binding["v_head_dim"], binding["max_q"],
    )


def config(binding, dtype):
    bs = binding["bs"]
    num_heads, v_head_dim, max_q = (
        binding["num_heads"], binding["v_head_dim"], binding["max_q"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "out": {"shape": ["total_q", num_heads, v_head_dim], "dtype": dt,
                    "note": "原地写回"},
            "lse": {"shape": ["total_q", num_heads], "dtype": "torch.float32",
                    "note": "原地写回"},
            "kv_lens": {"shape": [bs], "dtype": "torch.int32"},
            "cum_seq_lens": {"shape": [bs + 1], "dtype": "torch.int32"},
            "max_seq_len": {"note": "标量 int = max(q_lens)，数据相关"},
        },
        "outputs": {
            "out": {"shape": ["total_q", num_heads, v_head_dim], "dtype": dt},
            "lse": {"shape": ["total_q", num_heads], "dtype": "torch.float32"},
        },
        "dims": {
            "bs": bs, "num_heads": num_heads, "v_head_dim": v_head_dim, "max_q": max_q,
            "note": "total_q = sum(randint(1,max_q+1)) 数据相关（seed=11）",
        },
    }
