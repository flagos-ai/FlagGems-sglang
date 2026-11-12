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

"""Native SGLang collection workload for wide_row_softmax."""

import torch

OP_NAME = "wide_row_softmax"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (2,)

SHAPES = [(b, v) for b in (1, 4) for v in (32000, 131072)]
MORE_SHAPES = [(b, v) for b in (32,) for v in (32000, 131072)]


def native():
    """解析 wide_row_softmax_triton。"""
    try:
        from sglang.kernels.ops.speculative.multi_layer_eagle import (
            wide_row_softmax_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "vocab": vocab} for (bs, vocab) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, vocab = shape
    g = torch.Generator(device=device).manual_seed(9)
    logits = (
        torch.randn(bs, vocab, dtype=torch.float32, device=device, generator=g)
        * 4
    )
    out = torch.full((bs, vocab), -1.0, dtype=torch.float32, device=device)
    return logits.contiguous(), None, out


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["vocab"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["bs"], binding["vocab"]]


def config(binding, dtype):
    bs, vocab = binding["bs"], binding["vocab"]
    return {
        "inputs": {
            "logits": {"shape": [bs, vocab], "dtype": "torch.float32"},
            "temperatures": {"scalar": None},
            "out": {
                "shape": [bs, vocab],
                "dtype": "torch.float32",
                "note": "原地写回（out_stride=vocab）",
            },
        },
        "outputs": {
            "out": {"shape": [bs, vocab], "dtype": "torch.float32"},
        },
        "dims": {"bs": bs, "vocab": vocab, "out_stride": vocab},
    }
