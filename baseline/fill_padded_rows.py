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

"""Native SGLang collection workload for fill_padded_rows."""

import torch

OP_NAME = "fill_padded_rows"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(rows, 8, rows // 2) for rows in (8, 64, 512, 4096)]
MORE_SHAPES = [(rows, 8, rows // 2) for rows in (16384,)]


def native():
    try:
        from sglang.kernels.ops.moe.fill_padded_rows import (
            _fill_padded_rows as _f,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _op(out, num_token_non_padded, fill_value):
        _f(out, num_token_non_padded, fill_value)
        return out

    return _op


def grid():
    return [
        {"rows": rows, "cols": cols, "valid": mask_len}
        for (rows, cols, mask_len) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    rows, cols, mask_len = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        rows, cols, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    num_token_non_padded = torch.tensor(
        [mask_len], dtype=torch.int32, device=device
    )
    return (x, num_token_non_padded, -1.0)


def build_inputs(binding, dtype, device):
    shape = (binding["rows"], binding["cols"], binding["valid"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["rows"], binding["cols"]]


def config(binding, dtype):
    rows, cols, valid = binding["rows"], binding["cols"], binding["valid"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {
                "shape": [rows, cols],
                "dtype": dt,
                "note": "原地写回（x.clone()）",
            },
            "num_token_non_padded": {"shape": [1], "dtype": "torch.int32"},
            "fill_value": {"scalar": -1.0},
        },
        "outputs": {
            "x": {"shape": [rows, cols], "dtype": dt},
        },
        "dims": {"rows": rows, "cols": cols, "valid": valid},
    }
