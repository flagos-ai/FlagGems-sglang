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

"""Native SGLang collection workload for compute_position."""

import torch

OP_NAME = "compute_position"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(bs, 512) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 512) for bs in (1024,)]


def native():
    """解析 sglang.kernels.ops.attention.position.compute_position_triton。"""
    try:
        from sglang.kernels.ops.attention.position import (
            compute_position_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "max_len": max_len}
        for (bs, max_len) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, max_len = shape
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    prefix = torch.randint(
        0, 4096, (bs,), dtype=torch.int32, device=device, generator=g
    )
    return prefix, seq_lens, int(seq_lens.sum())


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["max_len"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["bs"]]


def config(binding, dtype):
    bs = binding["bs"]
    return {
        "inputs": {
            "extend_prefix_lens": {"shape": [bs], "dtype": "torch.int32"},
            "extend_seq_lens": {"shape": [bs], "dtype": "torch.int32"},
            "extend_seq_lens_sum": {
                "note": "标量 int = sum(extend_seq_lens)，数据相关"
            },
        },
        "outputs": {
            "positions": {
                "note": "positions 向量，长度=extend_seq_lens_sum，数据相关"
            },
            "extend_start_loc": {"shape": [bs], "dtype": "torch.int32"},
        },
        "dims": {"bs": bs, "max_len": binding["max_len"]},
    }
