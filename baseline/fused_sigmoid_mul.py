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

"""Native SGLang collection workload for fused_sigmoid_mul."""

import torch

OP_NAME = "fused_sigmoid_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 32, 128) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 32, 128) for t in (1024, 4096)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_sigmoid_mul as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": num_tokens,
            "num_heads": num_heads,
            "head_dim": head_dim,
            **{"kind": "strided"},
        }
        for (num_tokens, num_heads, head_dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    attn_output = torch.randn(
        num_tokens, num_heads * head_dim, device=device, generator=g
    ).to(cur_dtype)
    gate = torch.randn(
        num_tokens, num_heads, head_dim, device=device, generator=g
    ).to(cur_dtype)
    return attn_output, gate


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["num_heads"], binding["head_dim"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    if binding["kind"] == "flat":
        return [binding["bs"], binding["hidden"]]
    return "strided_bs{bs}_h{num_heads}_d{head_dim}".format(**binding)


def config(binding, dtype):
    bs = binding["bs"]
    dt = str(dtype)
    if binding["kind"] == "flat":
        hidden = binding["hidden"]
        return {
            "inputs": {
                "attn_output": {"shape": [bs, hidden], "dtype": dt},
                "gate": {"shape": [bs, hidden], "dtype": dt},
            },
            "outputs": {"out": {"shape": [bs, hidden], "dtype": dt}},
            "dims": {"kind": "flat", "bs": bs, "hidden": hidden},
        }
    nh, hd = binding["num_heads"], binding["head_dim"]
    return {
        "inputs": {
            "attn_output": {"shape": [bs, nh * hd], "dtype": dt},
            "gate": {"shape": [bs, nh, hd], "dtype": dt},
        },
        "outputs": {"out": {"shape": [bs, nh * hd], "dtype": dt}},
        "dims": {"kind": "strided", "bs": bs, "num_heads": nh, "head_dim": hd},
    }
