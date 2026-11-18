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

"""Native SGLang collection workload for hash_topk."""

import torch

OP_NAME = "hash_topk"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 256, 8, 1) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 256, 8, 1) for t in (4096,)]
_VOCAB = 32768


def native():
    try:
        from sglang.kernels.ops.moe.triton_hash_topk import (
            hash_topk_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "num_routed": num_routed,
            "topk_routed": topk_routed,
            "shared": shared,
            **{"vocab": _VOCAB},
        }
        for (num_tokens, num_routed, topk_routed, shared) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, num_routed, topk_routed, shared = shape
    g = torch.Generator(device=device).manual_seed(9)
    logits = torch.randn(
        num_tokens, num_routed, device=device, generator=g
    ).to(cur_dtype)
    tid2eid = torch.randint(
        0,
        num_routed,
        (_VOCAB, topk_routed),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    input_ids = torch.randint(
        0, _VOCAB, (num_tokens,), dtype=torch.int32, device=device, generator=g
    )
    # scoring_func is a str, which unpack_to_args_kwargs only forwards as part
    # of a kwargs dict.
    return (
        logits,
        input_ids,
        tid2eid,
        {
            "num_fused_shared_experts": shared,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sqrtsoftplus",
        },
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["num_routed"],
        binding["topk_routed"],
        binding["shared"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return (
        *args,
        kwargs["num_fused_shared_experts"],
        kwargs["routed_scaling_factor"],
        kwargs["scoring_func"],
    ), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_nr{binding['num_routed']}_tk{binding['topk_routed']}_V{binding['vocab']}"


def config(binding, dtype):
    t, num_routed, topk_routed, shared, vocab = (
        binding["num_tokens"],
        binding["num_routed"],
        binding["topk_routed"],
        binding["shared"],
        binding["vocab"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "router_logits": {"shape": [t, num_routed], "dtype": dt},
            "input_ids": {"shape": [t], "dtype": "torch.int32"},
            "tid2eid": {"shape": [vocab, topk_routed], "dtype": "torch.int32"},
            "num_fused_shared_experts": {"scalar": shared},
            "routed_scaling_factor": {"scalar": 2.5},
            "scoring_func": {"scalar": "sqrtsoftplus"},
        },
        "outputs": {
            "topk_weights": {
                "shape": [t, topk_routed + shared],
                "dtype": "torch.float32",
            },
            "topk_ids": {
                "shape": [t, topk_routed + shared],
                "dtype": "torch.int32",
            },
        },
        "dims": {
            "num_tokens": t,
            "num_routed": num_routed,
            "topk_routed": topk_routed,
            "shared": shared,
            "vocab": vocab,
        },
    }
