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

"""Native SGLang collection workload for fused_eh_norm."""

import torch

OP_NAME = "fused_eh_norm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (2048, 4096, 7168)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (2048, 4096, 7168)]


def native():
    """解析 sglang.kernels.ops.layernorm.fused_eh_norm.fused_eh_norm。"""
    try:
        from sglang.kernels.ops.layernorm.fused_eh_norm import (
            fused_eh_norm as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": num_tokens, "hidden": hidden}
        for (num_tokens, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, hidden = shape
    eps = 1e-6
    g = torch.Generator(device=device).manual_seed(0)
    inputs_embeds = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    previous_hidden = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    enorm_weight = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    hnorm_weight = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    return (
        inputs_embeds,
        previous_hidden,
        enorm_weight,
        hnorm_weight,
        {"eps": eps},
    )


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["hidden"])
    *args, kwargs = build_case(shape, dtype, device)
    return (*args, kwargs["eps"]), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "inputs_embeds": {"shape": [bs, hidden], "dtype": dt},
            "previous_hidden": {"shape": [bs, hidden], "dtype": dt},
            "enorm_weight": {"shape": [hidden], "dtype": dt},
            "hnorm_weight": {"shape": [hidden], "dtype": dt},
            "eps": {"scalar": 1e-6},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
