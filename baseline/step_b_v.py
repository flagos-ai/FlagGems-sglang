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

"""step_b_v baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.gemm.kv_b_lora_absorbed.step_b_v_fwd
    调用形态复刻 problems/lora/step_b_v/baseline.py：
        step_b_v_fwd(attn_lora_a, B_buf, batch_info, base_output.clone(),
                     qk_nope_head_dim, v_head_dim)
    kernel 把结果写回传入的 base_output（baseline 先 clone）。采集侧每次
    build_inputs 新建 base_output -> 对外无副作用。IS_INPLACE=True。

输入构造复刻 problems/lora/step_b_v/cases.py 的 BENCH_CASES：
    full_k = qk_nope_head_dim + v_head_dim
    attn_lora_a = randn(s, h, rank).to(dtype)
    B_buf       = randn(num_lora, h*full_k, rank).to(dtype)
    base_output = randn(s, h, v_head_dim).to(dtype)
    scalings = [0.5+0.25*i]; batch_info permutation="none"。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "step_b_v"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 把结果写回 base_output 实参

# BENCH_CASES: (seg_lens, num_lora, h, rank, qk_nope_head_dim, v_head_dim)
_CASES = [
    ([64] * 8, 4, 128, 32, 128, 128),
    ([256] * 4, 2, 128, 16, 64, 128),
]


def native():
    """解析 sglang.kernels.ops.gemm.kv_b_lora_absorbed.step_b_v_fwd。"""
    try:
        from sglang.kernels.ops.gemm.kv_b_lora_absorbed import step_b_v_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "h": h, "rank": r,
         "qk_nope_head_dim": qn, "v_head_dim": vhd}
        for (sl, nl, h, r, qn, vhd) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, h, rank = binding["num_lora"], binding["h"], binding["rank"]
    qk_nope_head_dim, v_head_dim = binding["qk_nope_head_dim"], binding["v_head_dim"]
    s = sum(seg_lens)
    full_k = qk_nope_head_dim + v_head_dim
    g = torch.Generator(device=device).manual_seed(0)
    attn_lora_a = torch.randn(
        s, h, rank, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    b_buf = torch.randn(
        num_lora, h * full_k, rank, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    base_output = torch.randn(
        s, h, v_head_dim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [rank] * num_lora, device,
        scalings=scalings, permutation="none",
    )
    # baseline 传 base_output.clone()；采集侧每次新建等价，直接传本地张量。
    return (attn_lora_a, b_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return (f"s{s}_L{binding['num_lora']}_h{binding['h']}_r{binding['rank']}"
            f"_qn{binding['qk_nope_head_dim']}_vhd{binding['v_head_dim']}")


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, h, rank = binding["num_lora"], binding["h"], binding["rank"]
    qk_nope_head_dim, v_head_dim = binding["qk_nope_head_dim"], binding["v_head_dim"]
    s = sum(seg_lens)
    full_k = qk_nope_head_dim + v_head_dim
    dt = str(dtype)
    return {
        "inputs": {
            "attn_lora_a": {"shape": [s, h, rank], "dtype": dt},
            "B_buf": {"shape": [num_lora, h * full_k, rank], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "base_output": {"shape": [s, h, v_head_dim], "dtype": dt,
                            "note": "原地写回"},
            "qk_nope_head_dim": {"scalar": qk_nope_head_dim},
            "v_head_dim": {"scalar": v_head_dim},
        },
        "outputs": {
            "base_output": {"shape": [s, h, v_head_dim], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s, "num_lora": num_lora, "h": h,
            "rank": rank, "qk_nope_head_dim": qk_nope_head_dim,
            "v_head_dim": v_head_dim, "full_k": full_k,
        },
    }
