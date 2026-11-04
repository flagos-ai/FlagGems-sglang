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

"""topk_sigmoid baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.moe_topk_sigmoid.topk_sigmoid（destination-passing）。
复刻 problems/moe/topk_sigmoid/baseline.py：w=topk_weights.clone(), ids=topk_ids.clone()，
调 topk_sigmoid(w, ids, gating_output, renormalize=..., routed_scaling_factor=...)，
原地写 w/ids 并返回。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, num_experts=256, k=8,
renormalize=True, scaling=1.0）：topk_weights=zeros(t,k).fp32; topk_ids=zeros(t,k).int32;
gating=randn(t,num_experts).fp32。IS_INPLACE=True（写回 w/ids）。
"""

import torch

OP_NAME = "topk_sigmoid"
DTYPES = [torch.float32]
IS_INPLACE = True

_NUM_EXPERTS = 256
_K = 8
_RENORM = True
_SCALE = 1.0
_TS = [1, 8, 64, 512, 4096]


def native():
    try:
        from sglang.kernels.ops.moe.moe_topk_sigmoid import topk_sigmoid as _f
    except Exception:
        return None

    def _op(w, ids, gating_output, renormalize, routed_scaling_factor):
        _f(w, ids, gating_output, renormalize=renormalize,
           routed_scaling_factor=routed_scaling_factor)
        return w, ids

    return _op


def grid():
    return [
        {"num_tokens": t, "num_experts": _NUM_EXPERTS, "k": _K} for t in _TS
    ]


def build_inputs(binding, dtype, device):
    t, num_experts, k = binding["num_tokens"], binding["num_experts"], binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    w = torch.zeros(t, k, dtype=torch.float32, device=device)
    ids = torch.zeros(t, k, dtype=torch.int32, device=device)
    gating = torch.randn(t, num_experts, device=device, generator=g)
    return (w, ids, gating, _RENORM, _SCALE), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_experts"], binding["k"]]


def config(binding, dtype):
    t, num_experts, k = binding["num_tokens"], binding["num_experts"], binding["k"]
    return {
        "inputs": {
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32", "note": "原地写回（clone）"},
            "topk_ids": {"shape": [t, k], "dtype": "torch.int32", "note": "原地写回（clone）"},
            "gating_output": {"shape": [t, num_experts], "dtype": "torch.float32"},
            "renormalize": {"scalar": _RENORM},
            "routed_scaling_factor": {"scalar": _SCALE},
        },
        "outputs": {
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "torch.int32"},
        },
        "dims": {"num_tokens": t, "num_experts": num_experts, "k": k},
    }
