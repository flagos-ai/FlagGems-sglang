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

"""Native SGLang collection workload for indexed_scale_shift."""

import torch

OP_NAME = "indexed_scale_shift"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(rows, 3072, 8) for rows in (1, 64, 1024, 8192)]
MORE_SHAPES = [(rows, 3072, 8) for rows in (32768,)]


def native():
    """解析 indexed_scale_shift_bf16_。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
            indexed_scale_shift_bf16_ as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
                indexed_scale_shift_bf16_ as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"rows": rows, "hidden": hidden, "variants": variants}
        for (rows, hidden, variants) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    rows, hidden, variants = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(rows, hidden, device=device, generator=g).to(cur_dtype)
    shift = torch.randn(variants, hidden, device=device, generator=g).to(
        cur_dtype
    )
    scale = torch.randn(variants, hidden, device=device, generator=g).to(
        cur_dtype
    )
    indices = torch.randint(
        0, variants, (rows,), dtype=torch.int32, device=device, generator=g
    )
    return x, shift, scale, indices


def build_inputs(binding, dtype, device):
    shape = (binding["rows"], binding["hidden"], binding["variants"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["rows"], binding["hidden"]]


def config(binding, dtype):
    rows, hidden, variants = (
        binding["rows"],
        binding["hidden"],
        binding["variants"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "out": {
                "shape": [rows, hidden],
                "dtype": dt,
                "note": "原地写回（x.clone()）",
            },
            "shift": {"shape": [variants, hidden], "dtype": dt},
            "scale": {"shape": [variants, hidden], "dtype": dt},
            "indices": {"shape": [rows], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [rows, hidden], "dtype": dt},
        },
        "dims": {"rows": rows, "hidden": hidden, "variants": variants},
    }
