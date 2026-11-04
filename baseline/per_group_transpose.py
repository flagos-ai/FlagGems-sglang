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

"""per_group_transpose baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.quantization.fp8_kernel.per_group_transpose
    调用形态复刻 problems/quantization/per_group_transpose/baseline.py：
        per_group_transpose(a, expert_offsets, m_alignment=1)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/quantization/per_group_transpose/cases.py 的 BENCH_CASES：
    counts=[n]*8；m=sum(counts)=8n；a=randn(m, k).bf16.contiguous()；
    expert_offsets=cumsum([0]+counts) int32。
dtype 取 benchmark/test_per_group_transpose.py 的 [bfloat16]。
"""

import torch

OP_NAME = "per_group_transpose"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: _case(k, [n]*8) for k in (128,512,4096) for n in (16,128,1024)
_SHAPES = [{"k": k, "n": n} for k in (128, 512, 4096) for n in (16, 128, 1024)]


def native():
    """解析 per_group_transpose。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import per_group_transpose as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    k, n = binding["k"], binding["n"]
    counts = [n] * 8
    m = sum(counts)
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, dtype=dtype, device=device, generator=g).contiguous()
    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)
    expert_offsets = torch.tensor(cum, dtype=torch.int32, device=device)
    return (a, expert_offsets, 1), {}


def key_shape(binding):
    n = binding["n"]
    return [8 * n, binding["k"]]


def config(binding, dtype):
    k, n = binding["k"], binding["n"]
    m = 8 * n
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [m, k], "dtype": dt},
            "expert_offsets": {"shape": [9], "dtype": "torch.int32",
                               "note": "cumsum([0]+[n]*8)，num_groups+1"},
            "m_alignment": {"scalar": 1},
        },
        "outputs": {
            "out": {"shape": [m, k], "dtype": dt, "note": "逐组转置结果"},
        },
        "dims": {"m": m, "k": k, "n_per_group": n, "num_groups": 8},
    }
