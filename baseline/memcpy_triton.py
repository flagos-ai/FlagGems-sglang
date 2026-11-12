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

"""Native SGLang collection workload for memcpy_triton."""

import torch

OP_NAME = "memcpy_triton"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(rows, 4096) for rows in (8, 64, 512, 4096)]
MORE_SHAPES = [(rows, 4096) for rows in (16384,)]


def native():
    """解析 sglang.kernels.ops.memory.memcpy_triton.memcpy_triton。"""
    try:
        from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "rows": rows,
            "hidden": hidden,
            **{"offset": 0, "sz": rows // 2, "offset_src": True},
        }
        for (rows, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    rows, hidden = shape
    g = torch.Generator(device=device).manual_seed(9)
    src = torch.randn(rows, hidden, device=device, generator=g).to(cur_dtype)
    dst = torch.zeros(rows, hidden, dtype=cur_dtype, device=device)
    # offset and size live in device memory so the copy stays capturable.
    offset = torch.tensor([0], dtype=torch.int32, device=device)
    sz = torch.tensor([rows // 2], dtype=torch.int32, device=device)
    return dst, src, 0, offset, sz, True


def build_inputs(binding, dtype, device):
    shape = (binding["rows"], binding["hidden"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"rows{binding['rows']}_h{binding['hidden']}"
        f"_sz{binding['sz']}_off{binding['offset']}"
    )


def config(binding, dtype):
    rows, hidden = binding["rows"], binding["hidden"]
    offset, sz, offset_src = (
        binding["offset"],
        binding["sz"],
        binding["offset_src"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "dst": {"shape": [rows, hidden], "dtype": dt, "note": "原地写回"},
            "src": {"shape": [rows, hidden], "dtype": dt},
            "dim": {"scalar": 0},
            "offset": {"shape": [1], "dtype": "torch.int32"},
            "sz": {"shape": [1], "dtype": "torch.int32"},
            "offset_src": {"scalar": offset_src},
        },
        "outputs": {
            "dst": {"shape": [rows, hidden], "dtype": dt},
        },
        "dims": {"rows": rows, "hidden": hidden, "offset": offset, "sz": sz},
    }
