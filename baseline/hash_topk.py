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

"""hash_topk baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.triton_hash_topk.hash_topk_triton。复刻
problems/moe/hash_topk/baseline.py：hash_topk_triton(router_logits, input_ids,
tid2eid, num_fused_shared_experts, routed_scaling_factor, scoring_func)，返回
新张量，不原地 -> IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, num_routed=256,
topk_routed=8, shared=1, vocab=32768, scaling=2.5, scoring="sqrtsoftplus"）：
    logits=randn(t,num_routed).bf16; tid2eid=randint(0,num_routed,(vocab,topk)).int32;
    input_ids=randint(0,vocab,(t,)).int32。
"""

import torch

OP_NAME = "hash_topk"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NUM_ROUTED = 256
_TOPK_ROUTED = 8
_SHARED = 1
_VOCAB = 32768
_SCALE = 2.5
_SCORING = "sqrtsoftplus"
_TS = [1, 8, 64, 512, 4096]


def native():
    try:
        from sglang.kernels.ops.moe.triton_hash_topk import hash_topk_triton as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "num_routed": _NUM_ROUTED, "topk_routed": _TOPK_ROUTED,
         "shared": _SHARED, "vocab": _VOCAB}
        for t in _TS
    ]


def build_inputs(binding, dtype, device):
    t, num_routed, topk_routed, shared, vocab = (
        binding["num_tokens"], binding["num_routed"], binding["topk_routed"],
        binding["shared"], binding["vocab"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(t, num_routed, device=device, generator=g).to(dtype)
    tid2eid = torch.randint(
        0, num_routed, (vocab, topk_routed), dtype=torch.int32, device=device, generator=g
    )
    input_ids = torch.randint(
        0, vocab, (t,), dtype=torch.int32, device=device, generator=g
    )
    return (logits, input_ids, tid2eid, shared, _SCALE, _SCORING), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_nr{binding['num_routed']}_tk{binding['topk_routed']}_V{binding['vocab']}"


def config(binding, dtype):
    t, num_routed, topk_routed, shared, vocab = (
        binding["num_tokens"], binding["num_routed"], binding["topk_routed"],
        binding["shared"], binding["vocab"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "router_logits": {"shape": [t, num_routed], "dtype": dt},
            "input_ids": {"shape": [t], "dtype": "torch.int32"},
            "tid2eid": {"shape": [vocab, topk_routed], "dtype": "torch.int32"},
            "num_fused_shared_experts": {"scalar": shared},
            "routed_scaling_factor": {"scalar": _SCALE},
            "scoring_func": {"scalar": _SCORING},
        },
        "outputs": {
            "topk_weights": {"shape": [t, topk_routed + shared], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, topk_routed + shared], "dtype": "torch.int32"},
        },
        "dims": {
            "num_tokens": t, "num_routed": num_routed, "topk_routed": topk_routed,
            "shared": shared, "vocab": vocab,
        },
    }
