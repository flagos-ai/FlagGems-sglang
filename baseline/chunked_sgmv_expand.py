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

"""chunked_sgmv_expand baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.gemm.chunked_sgmv_expand.chunked_sgmv_lora_expand_forward
    调用形态复刻 problems/lora/chunked_sgmv_expand/baseline.py：
        chunked_sgmv_lora_expand_forward(x, weights, batch_info, slice_offsets,
                                         max_slice_size, base_output.clone())
    kernel 把结果写回传入的 base_output（baseline 先 clone）-> IS_INPLACE=True。

输入构造复刻 problems/lora/chunked_sgmv_expand/cases.py 的 BENCH_CASES：
    n_slices = len(slice_sizes); output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)
    slice_offsets = int32 前缀和 [0, s0, s0+s1, ...]
    x = randn(s, n_slices*r).to(dtype)
    weights = randn(num_lora, output_dim, r).to(dtype)
    base_output = randn(s, output_dim).to(dtype)
    scalings = [0.5+0.25*i]; batch_info permutation="identity"。
dtype 取 benchmark/test_chunked_sgmv_expand.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "chunked_sgmv_expand"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 把结果写回 base_output 实参

# BENCH_CASES: (seg_lens, num_lora, r, slice_sizes)
_CASES = [
    ([64] * 8, 4, 32, [4096, 4096]),
    ([256] * 4, 2, 64, [4096, 1024, 1024]),
]


def native():
    """解析 chunked_sgmv_expand.chunked_sgmv_lora_expand_forward。"""
    try:
        from sglang.kernels.ops.gemm.chunked_sgmv_expand import (
            chunked_sgmv_lora_expand_forward as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "r": r, "slice_sizes": list(ss)}
        for (sl, nl, r, ss) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, r, slice_sizes = binding["num_lora"], binding["r"], binding["slice_sizes"]
    s = sum(seg_lens)
    n_slices = len(slice_sizes)
    output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)

    offsets = [0]
    for sz in slice_sizes:
        offsets.append(offsets[-1] + sz)
    slice_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        s, n_slices * r, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weights = torch.randn(
        num_lora, output_dim, r, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    base_output = torch.randn(
        s, output_dim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [r] * num_lora, device,
        scalings=scalings, permutation="identity",
    )
    return (x, weights, batch_info, slice_offsets, max_slice_size, base_output), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    ss = "x".join(str(z) for z in binding["slice_sizes"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_slc{ss}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, slice_sizes = (
        binding["num_lora"], binding["r"], binding["slice_sizes"],
    )
    s = sum(seg_lens)
    n_slices = len(slice_sizes)
    output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, n_slices * r], "dtype": dt},
            "weights": {"shape": [num_lora, output_dim, r], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "slice_offsets": {"shape": [n_slices + 1], "dtype": "torch.int32",
                              "note": "slice_sizes 的前缀和偏移"},
            "max_slice_size": {"scalar": max_slice_size},
            "base_output": {"shape": [s, output_dim], "dtype": dt,
                            "note": "原地写回"},
        },
        "outputs": {
            "base_output": {"shape": [s, output_dim], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s,
            "num_lora": num_lora, "r": r,
            "slice_sizes": list(slice_sizes), "n_slices": n_slices,
            "output_dim": output_dim, "max_slice_size": max_slice_size,
        },
    }
