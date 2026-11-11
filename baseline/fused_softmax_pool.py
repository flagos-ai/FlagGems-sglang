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

"""Native SGLang collection workload for fused_softmax_pool."""

import torch

OP_NAME = "fused_softmax_pool"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(bs, k, 128) for bs in (1, 64, 1024) for k in (4, 16, 64)]
MORE_SHAPES = [(bs, k, 128) for bs in (4096,) for k in (4, 16, 64)]


def native():
    """解析 deepseek_v4_rope.fused_softmax_pool_triton。"""
    try:
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            fused_softmax_pool_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "k": k, "head_dim": head_dim}
        for (bs, k, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, k, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    # The last axis packs [kv(head_dim), score(head_dim)].
    kv_score = torch.randn(bs, k, 2 * head_dim, device=device, generator=g).to(
        cur_dtype
    )
    return kv_score.contiguous(), head_dim


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["k"], binding["head_dim"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["bs"], binding["k"], 2 * binding["head_dim"]]


def config(binding, dtype):
    bs, k, head_dim = binding["bs"], binding["k"], binding["head_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "kv_score": {"shape": [bs, k, 2 * head_dim], "dtype": dt},
            "head_dim": {"scalar": head_dim},
        },
        "outputs": {
            "out": {"shape": [bs, k, head_dim], "dtype": dt},
        },
        "dims": {"bs": bs, "k": k, "head_dim": head_dim},
    }
