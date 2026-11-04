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

"""grouped_gemm baseline（方案 B，sglang）。

native（注意：problem 属 gemm 组，但符号在 moe.inkling_moe 下）：
    sglang.kernels.ops.moe.inkling_moe.grouped_gemm_triton
    调用形态复刻 problems/gemm/grouped_gemm/baseline.py：
        grouped_gemm_triton(a, b, num_experts, num_tokens_per_expert,
                            expert_token_offs, expert_block_offs,
                            expert_block_schedule)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/gemm/grouped_gemm/cases.py 的 BENCH_CASES：
    a = randn(m, k).bf16; b = (0.1*randn(num_experts, n, k)).bf16
    ids = randint(0, num_experts, (m,)) int32，排序后用 sglang 的
    compute_grouped_gemm_metadata 产出分组 metadata（运行时 import，self-contained）。
    其中 n=512, k=2048，m in (128,1024,4096)，num_experts in (8,32)。
metadata 为数据相关张量，key_shape 用语义字符串。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "grouped_gemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: (m, n, k, num_experts)；n=512, k=2048
_CASES = [
    (m, 512, 2048, e) for m in (128, 1024, 4096) for e in (8, 32)
]


def native():
    """解析 sglang.kernels.ops.moe.inkling_moe.grouped_gemm_triton。"""
    try:
        from sglang.kernels.ops.moe.inkling_moe import grouped_gemm_triton as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "n": n, "k": k, "num_experts": e}
        for (m, n, k, e) in _CASES
    ]


def build_inputs(binding, dtype, device):
    from sglang.kernels.ops.moe.inkling_moe import compute_grouped_gemm_metadata

    m, n, k, num_experts = (
        binding["m"], binding["n"], binding["k"], binding["num_experts"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float32, device=device, generator=g).to(dtype)
    b = (
        0.1 * torch.randn(num_experts, n, k, dtype=torch.float32, device=device, generator=g)
    ).to(dtype)
    ids = torch.randint(
        0, num_experts, (m,), device=device, generator=g, dtype=torch.int32
    )
    sorted_ids = torch.sort(ids)[0].contiguous()
    (
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
    ) = compute_grouped_gemm_metadata(sorted_ids, num_experts)
    return (
        a.contiguous(),
        b.contiguous(),
        num_experts,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
    ), {}


def key_shape(binding):
    return (f"m{binding['m']}_n{binding['n']}_k{binding['k']}"
            f"_E{binding['num_experts']}")


def config(binding, dtype):
    m, n, k, num_experts = (
        binding["m"], binding["n"], binding["k"], binding["num_experts"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [m, k], "dtype": dt},
            "b": {"shape": [num_experts, n, k], "dtype": dt},
            "num_experts": {"scalar": num_experts},
            "num_tokens_per_expert": {"note": "compute_grouped_gemm_metadata 产出，"
                                      "shape [num_experts]，数据相关"},
            "expert_token_offs": {"note": "分组 token 偏移，数据相关"},
            "expert_block_offs": {"note": "分组 block 偏移，数据相关"},
            "expert_block_schedule": {"note": "block 调度表，数据相关"},
        },
        "outputs": {
            "c": {"shape": [m, n], "dtype": dt},
        },
        "dims": {
            "m": m, "n": n, "k": k, "num_experts": num_experts,
            "note": "metadata 由 sorted expert ids 经 compute_grouped_gemm_metadata 产出，数据相关",
        },
    }
