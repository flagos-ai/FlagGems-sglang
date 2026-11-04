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

"""qkv_lora_b baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.gemm.qkv_lora_b.qkv_lora_b_fwd
    调用形态复刻 problems/lora/qkv_lora_b/baseline.py：
        qkv_lora_b_fwd(x, qkv_lora_b, batch_info, output_offset,
                       max_qkv_out_dim, base_output.clone())
    kernel 把结果写回传入的 base_output（baseline 先 clone）-> IS_INPLACE=True。

输入构造复刻 problems/lora/qkv_lora_b/cases.py 的 BENCH_CASES：
    output_dim = dq + 2*dkv; max_qkv_out_dim = max(dq, dkv)
    x = randn(s, 3*r).to(dtype)
    qkv_lora_b = randn(num_lora, output_dim, r).to(dtype)
    base_output = randn(s, output_dim).to(dtype)
    output_offset = int32[0, dq, dq+dkv, dq+2*dkv]
    scalings = [0.5+0.25*i]; batch_info permutation="none"。
dtype 取 benchmark/test_qkv_lora_b.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "qkv_lora_b"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 把结果写回 base_output 实参

# BENCH_CASES: (seg_lens, num_lora, r, dq, dkv)
_CASES = [
    ([64] * 8, 4, 32, 4096, 1024),
    ([256] * 4, 2, 64, 4096, 1024),
]


def native():
    """解析 sglang.kernels.ops.gemm.qkv_lora_b.qkv_lora_b_fwd。"""
    try:
        from sglang.kernels.ops.gemm.qkv_lora_b import qkv_lora_b_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "r": r, "dq": dq, "dkv": dkv}
        for (sl, nl, r, dq, dkv) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, r, dq, dkv = (
        binding["num_lora"], binding["r"], binding["dq"], binding["dkv"],
    )
    s = sum(seg_lens)
    output_dim = dq + 2 * dkv
    max_qkv_out_dim = max(dq, dkv)
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(s, 3 * r, generator=g, device=device, dtype=torch.float32).to(dtype)
    qkv_lora_b = torch.randn(
        num_lora, output_dim, r, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    base_output = torch.randn(
        s, output_dim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    output_offset = torch.tensor(
        [0, dq, dq + dkv, dq + 2 * dkv], dtype=torch.int32, device=device
    )
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [r] * num_lora, device,
        scalings=scalings, permutation="none",
    )
    return (x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_dq{binding['dq']}_dkv{binding['dkv']}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, dq, dkv = (
        binding["num_lora"], binding["r"], binding["dq"], binding["dkv"],
    )
    s = sum(seg_lens)
    output_dim = dq + 2 * dkv
    max_qkv_out_dim = max(dq, dkv)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, 3 * r], "dtype": dt},
            "qkv_lora_b": {"shape": [num_lora, output_dim, r], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "output_offset": {"shape": [4], "dtype": "torch.int32",
                              "note": "[0, dq, dq+dkv, dq+2*dkv] 的 qkv 切片偏移"},
            "max_qkv_out_dim": {"scalar": max_qkv_out_dim},
            "base_output": {"shape": [s, output_dim], "dtype": dt,
                            "note": "原地写回"},
        },
        "outputs": {
            "base_output": {"shape": [s, output_dim], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s,
            "num_lora": num_lora, "r": r, "dq": dq, "dkv": dkv,
            "output_dim": output_dim, "max_qkv_out_dim": max_qkv_out_dim,
        },
    }
