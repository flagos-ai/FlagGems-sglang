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

"""per_token_group_quant_fp8_moe_mn_major baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_kernel.
        per_token_group_quant_fp8_hopper_moe_mn_major
    调用形态复刻 baseline.py：
        per_token_group_quant_fp8_hopper_moe_mn_major(
            A, expert_offsets, problem_sizes, group_size)
    返回 (a_q, sfa) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（group_size=128，各 expert 行数相同）：
    m_total=sum(rows_per_expert)；A=randn(m_total,k).to(bf16)；
    expert_offsets=cumsum int32 (E+1,)；problem_sizes=(E,3) int32 [rows,k,k]。
BENCH: _case([n]*e,k) for e,n in ((8,256),(32,128)) for k in (2048,4096,7168)。
key_shape 用语义字符串（含 expert 列表派生维度）。
"""

import torch

OP_NAME = "per_token_group_quant_fp8_moe_mn_major"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_GROUP = 128
_CASES = [
    ([n] * e, k)
    for e, n in ((8, 256), (32, 128))
    for k in (2048, 4096, 7168)
]


def native():
    """解析 per_token_group_quant_fp8_hopper_moe_mn_major。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8_hopper_moe_mn_major as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"rows_per_expert": list(rpe), "k": k, "group_size": _GROUP}
        for (rpe, k) in _CASES
    ]


def build_inputs(binding, dtype, device):
    rows_per_expert = binding["rows_per_expert"]
    k = binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    m_total = sum(rows_per_expert)
    A = torch.randn(m_total, k, dtype=torch.float32, device=device, generator=g).to(dtype)
    offs = torch.tensor(
        [0] + list(torch.tensor(rows_per_expert).cumsum(0)),
        dtype=torch.int32,
        device=device,
    )
    problem_sizes = torch.zeros(
        (len(rows_per_expert), 3), dtype=torch.int32, device=device
    )
    problem_sizes[:, 0] = torch.tensor(rows_per_expert, dtype=torch.int32, device=device)
    problem_sizes[:, 1] = k
    problem_sizes[:, 2] = k
    return (A.contiguous(), offs, problem_sizes, binding["group_size"]), {}


def key_shape(binding):
    e = len(binding["rows_per_expert"])
    m = sum(binding["rows_per_expert"])
    return f"E{e}_m{m}_k{binding['k']}"


def config(binding, dtype):
    rows_per_expert = binding["rows_per_expert"]
    k = binding["k"]
    e = len(rows_per_expert)
    m_total = sum(rows_per_expert)
    dt = str(dtype)
    return {
        "inputs": {
            "A": {"shape": [m_total, k], "dtype": dt},
            "expert_offsets": {"shape": [e + 1], "dtype": "torch.int32"},
            "problem_sizes": {"shape": [e, 3], "dtype": "torch.int32"},
            "group_size": {"scalar": _GROUP},
        },
        "outputs": {
            "a_q": {"shape": [m_total, k], "dtype": "torch.float8_e4m3fn"},
            "sfa": {"shape": [m_total, k // _GROUP], "dtype": "torch.float32"},
        },
        "dims": {
            "rows_per_expert": list(rows_per_expert),
            "num_experts": e, "m_total": m_total, "k": k, "group_size": _GROUP,
        },
    }
