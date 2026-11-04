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

"""zero_experts_identity baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.zero_experts_compute_triton。复刻
problems/moe/zero_experts_identity/baseline.py：
    zero_experts_compute_triton(expert_indices.clone(), expert_scales.clone(),
        num_experts, zero_expert_type, hidden_states)
前两个入参 clone 后传入（kernel 可能原地改写索引/scale）-> IS_INPLACE=True。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, top_k=8, hidden=7168,
num_experts=32, num_zero=8, zero_expert_type="identity"）：
    idx=randint(0,num_experts+num_zero,(t,top_k)).int32; scales=rand(t,top_k).fp32;
    hidden_states=randn(t,hidden).bf16。
"""

import torch

OP_NAME = "zero_experts_identity"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_TOP_K = 8
_NUM_EXPERTS = 32
_NUM_ZERO = 8
_ZERO_TYPE = "identity"
_SHAPES = [(t, 7168) for t in (1, 8, 64, 512, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            zero_experts_compute_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "top_k": _TOP_K, "hidden": h,
         "num_experts": _NUM_EXPERTS, "num_zero": _NUM_ZERO}
        for (t, h) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t, top_k, hidden, num_experts, num_zero = (
        binding["num_tokens"], binding["top_k"], binding["hidden"],
        binding["num_experts"], binding["num_zero"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    idx = torch.randint(
        0, num_experts + num_zero, (t, top_k), dtype=torch.int32, device=device, generator=g
    )
    scales = torch.rand(t, top_k, dtype=torch.float32, device=device, generator=g)
    h = torch.randn(t, hidden, device=device, generator=g).to(dtype)
    # baseline 对前两者 clone 后传入。
    return (idx.clone(), scales.clone(), num_experts, _ZERO_TYPE, h), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_tk{binding['top_k']}_h{binding['hidden']}_E{binding['num_experts']}_z{binding['num_zero']}"


def config(binding, dtype):
    t, top_k, hidden, num_experts, num_zero = (
        binding["num_tokens"], binding["top_k"], binding["hidden"],
        binding["num_experts"], binding["num_zero"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "expert_indices": {"shape": [t, top_k], "dtype": "torch.int32", "note": "原地写回（clone）"},
            "expert_scales": {"shape": [t, top_k], "dtype": "torch.float32", "note": "原地写回（clone）"},
            "num_experts": {"scalar": num_experts},
            "zero_expert_type": {"scalar": _ZERO_TYPE},
            "hidden_states": {"shape": [t, hidden], "dtype": dt},
        },
        "outputs": {
            "hidden_states_out": {"shape": [t, hidden], "dtype": dt},
        },
        "dims": {"num_tokens": t, "top_k": top_k, "hidden": hidden,
                 "num_experts": num_experts, "num_zero": num_zero},
    }
