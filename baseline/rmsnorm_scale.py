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

"""Native SGLang collection workload for rmsnorm_scale."""

import torch

OP_NAME = "rmsnorm_scale"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(r, d) for r in (8, 64, 1024) for d in (2048, 4096)]
MORE_SHAPES = [(8192, d) for d in (2048, 4096)]
EPS = 1e-6


def native():
    """解析 rmsnorm_scale。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.native_bf16_rmsnorm import (
            rmsnorm_scale as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
                rmsnorm_scale as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"rows": rows, "dim": dim} for (rows, dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    rows, dim = binding["rows"], binding["dim"]
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    # scale has one row per x row here; the broadcast path is covered by the
    # test.
    return (_t(rows, dim), _t(dim), _t(rows, dim), EPS), {}


def key_shape(binding):
    return [binding["rows"], binding["dim"]]


def config(binding, dtype):
    rows, dim = binding["rows"], binding["dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [rows, dim], "dtype": dt},
            "weight": {"shape": [dim], "dtype": dt},
            "scale": {"shape": [rows, dim], "dtype": dt},
            "eps": {"scalar": EPS},
        },
        "outputs": {
            "out": {"shape": [rows, dim], "dtype": dt},
        },
        "dims": {"rows": rows, "dim": dim},
    }
