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

"""chunked_sgmv_shrink baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.gemm.chunked_sgmv_shrink.chunked_sgmv_lora_shrink_forward
    调用形态复刻 problems/lora/chunked_sgmv_shrink/baseline.py：
        chunked_sgmv_lora_shrink_forward(x, weights, batch_info, num_slices)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/lora/chunked_sgmv_shrink/cases.py 的 BENCH_CASES：
    x = randn(s, k).to(dtype); weights = randn(num_lora, num_slices*r, k).to(dtype)
    batch_info permutation="identity"（该 kernel 恒需显式置换）。
dtype 取 benchmark/test_chunked_sgmv_shrink.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "chunked_sgmv_shrink"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: (seg_lens, num_lora, r, k, num_slices)
_CASES = [
    ([64] * 8, 4, 32, 4096, 1),
    ([256] * 4, 2, 64, 4096, 1),
]


def native():
    """解析 chunked_sgmv_shrink.chunked_sgmv_lora_shrink_forward。"""
    try:
        from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
            chunked_sgmv_lora_shrink_forward as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "r": r, "k": k, "num_slices": ns}
        for (sl, nl, r, k, ns) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, r, k, num_slices = (
        binding["num_lora"], binding["r"], binding["k"], binding["num_slices"],
    )
    s = sum(seg_lens)
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(s, k, generator=g, device=device, dtype=torch.float32).to(dtype)
    weights = torch.randn(
        num_lora, num_slices * r, k, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [r] * num_lora, device, permutation="identity"
    )
    return (x, weights, batch_info, num_slices), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return (f"s{s}_L{binding['num_lora']}_r{binding['r']}_k{binding['k']}"
            f"_sl{binding['num_slices']}")


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, k, num_slices = (
        binding["num_lora"], binding["r"], binding["k"], binding["num_slices"],
    )
    s = sum(seg_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, k], "dtype": dt},
            "weights": {"shape": [num_lora, num_slices * r, k], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "num_slices": {"scalar": num_slices},
        },
        "outputs": {
            "output": {"shape": [s, num_slices * r], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s,
            "num_lora": num_lora, "r": r, "k": k, "num_slices": num_slices,
        },
    }
