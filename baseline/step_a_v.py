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

"""step_a_v baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.gemm.kv_b_lora_absorbed.step_a_v_fwd
    调用形态复刻 problems/lora/step_a_v/baseline.py：
        step_a_v_fwd(attn_output, A_buf, batch_info)
    产出新张量返回，不写回入参 -> IS_INPLACE=False。

输入构造复刻 problems/lora/step_a_v/cases.py 的 BENCH_CASES：
    attn_output = randn(s, h, kv_lora_rank).to(dtype)
    A_buf       = randn(num_lora, rank, kv_lora_rank).to(dtype)
    batch_info  = make_batch_info(seg_lens, weight_indices, [rank]*num_lora,
                                  permutation="none")
    其中 s = sum(seg_lens), weight_indices = [i % num_lora for ...]。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "step_a_v"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: (seg_lens, num_lora, h, kv_lora_rank, rank)
_CASES = [
    ([64] * 8, 4, 128, 512, 32),
    ([256] * 4, 2, 128, 512, 16),
]


def native():
    """解析 sglang.kernels.ops.gemm.kv_b_lora_absorbed.step_a_v_fwd。"""
    try:
        from sglang.kernels.ops.gemm.kv_b_lora_absorbed import step_a_v_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "h": h, "kv_lora_rank": klr, "rank": r}
        for (sl, nl, h, klr, r) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, h = binding["num_lora"], binding["h"]
    kv_lora_rank, rank = binding["kv_lora_rank"], binding["rank"]
    s = sum(seg_lens)
    g = torch.Generator(device=device).manual_seed(0)
    attn_output = torch.randn(
        s, h, kv_lora_rank, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    a_buf = torch.randn(
        num_lora, rank, kv_lora_rank, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [rank] * num_lora, device, permutation="none"
    )
    return (attn_output, a_buf, batch_info), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return (f"s{s}_L{binding['num_lora']}_h{binding['h']}"
            f"_klr{binding['kv_lora_rank']}_r{binding['rank']}")


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, h = binding["num_lora"], binding["h"]
    kv_lora_rank, rank = binding["kv_lora_rank"], binding["rank"]
    s = sum(seg_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "attn_output": {"shape": [s, h, kv_lora_rank], "dtype": dt},
            "A_buf": {"shape": [num_lora, rank, kv_lora_rank], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
        },
        "outputs": {
            "output": {"shape": [s, h, rank], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s, "num_lora": num_lora, "h": h,
            "kv_lora_rank": kv_lora_rank, "rank": rank,
        },
    }
