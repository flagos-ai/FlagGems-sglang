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

"""Native SGLang collection workload for clamp_position."""

import torch

OP_NAME = "clamp_position"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(bs,) for bs in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(bs,) for bs in (16384,)]


def native():
    """解析 sglang.kernels.ops.attention.clamp_position.clamp_position_cuda。"""
    try:
        from sglang.kernels.ops.attention.clamp_position import (
            clamp_position_cuda as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs} for (bs,) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    (bs,) = shape
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = torch.randint(
        0, 8192, (bs,), dtype=torch.int32, device=device, generator=g
    )
    return (seq_lens,)


def build_inputs(binding, dtype, device):
    shape = (binding["bs"],)
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["bs"]]


def config(binding, dtype):
    bs = binding["bs"]
    return {
        "inputs": {
            "seq_lens": {"shape": [bs], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [bs], "dtype": "torch.int32"},
        },
        "dims": {"bs": bs},
    }
