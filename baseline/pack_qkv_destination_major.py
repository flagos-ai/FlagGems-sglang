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

"""Native SGLang collection workload for pack_qkv_destination_major."""

import torch

OP_NAME = "pack_qkv_destination_major"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(r, 32, 128, 8) for r in (64, 1024, 4096)]
MORE_SHAPES = [(16384, 32, 128, 8)]


def native():
    """解析 pack_qkv_destination_major。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
            pack_qkv_destination_major as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton import (
                pack_qkv_destination_major as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {
            "rows": rows,
            "global_heads": global_heads,
            "head_size": head_size,
            "world_size": world_size,
        }
        for (rows, global_heads, head_size, world_size) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_inputs(binding, dtype, device):
    rows, global_heads, head_size, world_size = (
        binding["rows"],
        binding["global_heads"],
        binding["head_size"],
        binding["world_size"],
    )
    g = torch.Generator(device=device).manual_seed(9)

    def _t():
        return torch.randn(
            rows,
            global_heads,
            head_size,
            dtype=torch.float32,
            device=device,
            generator=g,
        ).to(dtype)

    # out=None lets the operator allocate; the passed-out path is covered by
    # the test.
    return (_t(), _t(), _t(), world_size, None), {}


def key_shape(binding):
    r, gh = binding["rows"], binding["global_heads"]
    return f"r{r}_gh{gh}_hs{binding['head_size']}_ws{binding['world_size']}"


def config(binding, dtype):
    rows, global_heads = binding["rows"], binding["global_heads"]
    head_size, world_size = binding["head_size"], binding["world_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [rows, global_heads, head_size], "dtype": dt},
            "k": {"shape": [rows, global_heads, head_size], "dtype": dt},
            "v": {"shape": [rows, global_heads, head_size], "dtype": dt},
            "world_size": {"scalar": world_size},
            "out": {"scalar": None},
        },
        "outputs": {
            "out": {
                "shape": [
                    world_size,
                    rows,
                    global_heads // world_size,
                    3 * head_size,
                ],
                "dtype": dt,
            },
        },
        "dims": {
            "rows": rows,
            "global_heads": global_heads,
            "head_size": head_size,
            "world_size": world_size,
        },
    }
