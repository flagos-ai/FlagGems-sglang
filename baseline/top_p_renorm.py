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

"""Native SGLang collection workload for top_p_renorm."""

import torch

OP_NAME = "top_p_renorm"
DTYPES = [torch.float32]
IS_INPLACE = False

SHAPES = [(b, v) for b in (1, 8, 64) for v in (32000, 131072)]
MORE_SHAPES = [(b, v) for b in (256,) for v in (32000, 131072)]
_TOP_P = 0.9


def native():
    """解析 top_p_renorm_probs_triton。"""
    try:
        from sglang.kernels.ops.sampling.top_p_renorm_triton import (
            top_p_renorm_probs_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": batch, "vocab": vocab}
        for (batch, vocab) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, vocab = shape
    g = torch.Generator(device=device).manual_seed(9)
    logits = torch.randn(
        batch, vocab, dtype=torch.float32, device=device, generator=g
    )
    return torch.softmax(logits, dim=-1).contiguous(), _TOP_P


def build_inputs(binding, dtype, device):
    shape = (binding["batch"], binding["vocab"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["batch"], binding["vocab"]]


def config(binding, dtype):
    batch, vocab = binding["batch"], binding["vocab"]
    return {
        "inputs": {
            "probs": {"shape": [batch, vocab], "dtype": "torch.float32"},
            "top_p": {"scalar": _TOP_P},
        },
        "outputs": {
            "out": {"shape": [batch, vocab], "dtype": "torch.float32"},
        },
        "dims": {"batch": batch, "vocab": vocab},
    }
