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

"""step_a_q baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.gemm.kv_b_lora_absorbed.step_a_q_fwd
    调用形态复刻 problems/lora/step_a_q/baseline.py：
        step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    产出新张量返回，不写回入参 -> IS_INPLACE=False。

输入构造复刻 problems/lora/step_a_q/cases.py 的 BENCH_CASES：
    full_k = qk_nope + v_head_dim
    q_nope = randn(s, h, qk_nope).to(dtype)
    B_buf  = randn(num_lora, h*full_k, rank).to(dtype)
    batch_info = make_batch_info(seg_lens, weight_indices, [rank]*num_lora,
                                 permutation="none")
    其中 s = sum(seg_lens), weight_indices = [i % num_lora for ...]。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "step_a_q"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: (seg_lens, num_lora, h, qk_nope, v_head_dim, rank)
_CASES = [
    ([64] * 8, 4, 128, 128, 128, 32),
    ([256] * 4, 2, 128, 64, 128, 16),
]


def native():
    """解析 sglang.kernels.ops.gemm.kv_b_lora_absorbed.step_a_q_fwd。"""
    try:
        from sglang.kernels.ops.gemm.kv_b_lora_absorbed import step_a_q_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "h": h,
         "qk_nope": qn, "v_head_dim": vhd, "rank": r}
        for (sl, nl, h, qn, vhd, r) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, h = binding["num_lora"], binding["h"]
    qk_nope, v_head_dim, rank = binding["qk_nope"], binding["v_head_dim"], binding["rank"]
    s = sum(seg_lens)
    full_k = qk_nope + v_head_dim
    g = torch.Generator(device=device).manual_seed(0)
    q_nope = torch.randn(
        s, h, qk_nope, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    b_buf = torch.randn(
        num_lora, h * full_k, rank, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [rank] * num_lora, device, permutation="none"
    )
    return (q_nope, b_buf, batch_info, full_k), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return (f"s{s}_L{binding['num_lora']}_h{binding['h']}"
            f"_qn{binding['qk_nope']}_vhd{binding['v_head_dim']}_r{binding['rank']}")


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, h = binding["num_lora"], binding["h"]
    qk_nope, v_head_dim, rank = binding["qk_nope"], binding["v_head_dim"], binding["rank"]
    s = sum(seg_lens)
    full_k = qk_nope + v_head_dim
    dt = str(dtype)
    return {
        "inputs": {
            "q_nope": {"shape": [s, h, qk_nope], "dtype": dt},
            "B_buf": {"shape": [num_lora, h * full_k, rank], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "full_K_per_head": {"scalar": full_k},
        },
        "outputs": {
            "output": {"shape": [s, h, full_k], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s, "num_lora": num_lora, "h": h,
            "qk_nope": qk_nope, "v_head_dim": v_head_dim, "full_k": full_k, "rank": rank,
        },
    }
