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

"""fused_append_shared_experts baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.fused_moe_triton_kernels.fused_append_shared_experts。
复刻 problems/moe/fused_append_shared_experts/baseline.py：
    op(topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N=N)
返回新的 (topk_ids, topk_weights)，不原地写回输入 -> IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（m∈{1,8,64,512,4096,16384}, k=8, s=1,
n_base=256, scale=0.25）：topk_ids=randint(0,256,(m,k)).int32;
topk_weights=rand(m,k).fp32。
"""

import torch

OP_NAME = "fused_append_shared_experts"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_K = 8
_N_BASE = 256
_SCALE = 0.25
_MS = [1, 8, 64, 512, 4096, 16384]


def native():
    try:
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            fused_append_shared_experts as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "k": _K, "s": 1, "n_base": _N_BASE} for m in _MS]


def build_inputs(binding, dtype, device):
    m, k, s, n_base = binding["m"], binding["k"], binding["s"], binding["n_base"]
    g = torch.Generator(device=device).manual_seed(0)
    topk_ids = torch.randint(
        0, n_base, (m, k), device=device, generator=g, dtype=torch.int32
    ).contiguous()
    topk_weights = torch.rand(
        m, k, dtype=torch.float32, device=device, generator=g
    ).contiguous()
    return (topk_ids, topk_weights, s, _SCALE), {"N": n_base}


def key_shape(binding):
    return f"m{binding['m']}_k{binding['k']}_s{binding['s']}_N{binding['n_base']}"


def config(binding, dtype):
    m, k, s, n_base = binding["m"], binding["k"], binding["s"], binding["n_base"]
    return {
        "inputs": {
            "topk_ids": {"shape": [m, k], "dtype": "torch.int32"},
            "topk_weights": {"shape": [m, k], "dtype": "torch.float32"},
            "num_fused_shared_experts": {"scalar": s},
            "scale_factor": {"scalar": _SCALE},
            "N": {"scalar": n_base},
        },
        "outputs": {
            "topk_ids": {"shape": [m, k + s], "dtype": "torch.int32"},
            "topk_weights": {"shape": [m, k + s], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "k": k, "num_fused_shared_experts": s, "N": n_base},
    }
