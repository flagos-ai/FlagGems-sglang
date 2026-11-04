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

"""sgemm_lora_b baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.gemm.sgemm_lora_b.sgemm_lora_b_fwd
    调用形态复刻 problems/lora/sgemm_lora_b/baseline.py：
        sgemm_lora_b_fwd(x, weights, batch_info, base_output.clone())
    kernel 把结果累加写回传入的 base_output（故 baseline 先 clone）。采集侧
    每次 build_inputs 新建 base_output，等价于那次 clone，被写回的是本地新张量
    -> 对外部无副作用。IS_INPLACE=True：kernel 原地写回 base_output 实参，
    NCU 采样前应重建实参避免 warmup 累积污染。

输入构造复刻 problems/lora/sgemm_lora_b/cases.py 的 BENCH_CASES：
    x = randn(s, r).to(dtype); weights = randn(num_lora, n, r).to(dtype)
    base_output = randn(s, n).to(dtype)
    scalings = [0.5+0.25*i]; batch_info permutation="none"。
dtype 取 benchmark/test_sgemm_lora_b.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "sgemm_lora_b"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 把结果写回 base_output 实参

# BENCH_CASES: (seg_lens, num_lora, r, n)
_CASES = [
    ([64] * 8, 4, 32, 4096),
    ([256] * 4, 2, 64, 4096),
]


def native():
    """解析 sglang.kernels.ops.gemm.sgemm_lora_b.sgemm_lora_b_fwd。"""
    try:
        from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "r": r, "n": n}
        for (sl, nl, r, n) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, r, n = binding["num_lora"], binding["r"], binding["n"]
    s = sum(seg_lens)
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(s, r, generator=g, device=device, dtype=torch.float32).to(dtype)
    weights = torch.randn(
        num_lora, n, r, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    base_output = torch.randn(
        s, n, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [r] * num_lora, device,
        scalings=scalings, permutation="none",
    )
    # baseline 传 base_output.clone()；采集侧每次新建等价，直接传本地张量。
    return (x, weights, batch_info, base_output), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_n{binding['n']}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, n = binding["num_lora"], binding["r"], binding["n"]
    s = sum(seg_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, r], "dtype": dt},
            "weights": {"shape": [num_lora, n, r], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "base_output": {"shape": [s, n], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "base_output": {"shape": [s, n], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s,
            "num_lora": num_lora, "r": r, "n": n,
        },
    }
