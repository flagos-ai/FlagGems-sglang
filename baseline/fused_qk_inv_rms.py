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

"""Native SGLang collection workload for fused_qk_inv_rms."""

import torch

OP_NAME = "fused_qk_inv_rms"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(1, n, 16, 128) for n in (64, 512, 4096)]
MORE_SHAPES = [(1, 16384, 16, 128)]
_EPS = 1e-6


def native():
    """解析 fused_qk_inv_rms。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.attention.sana_wm_gdn_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.sana_wm_gdn import (
            fused_qk_inv_rms as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.attention.sana_wm_gdn_triton import (
                fused_qk_inv_rms as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "n": n, "heads": heads, "dim": dim}
        for (b, n, heads, dim) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    b, n, heads, dim = shape
    g = torch.Generator(device=device).manual_seed(9)
    qkv = torch.randn(
        b, n, 3, heads, dim, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    return qkv, _EPS


def build_inputs(binding, dtype, device):
    shape = (binding["b"], binding["n"], binding["heads"], binding["dim"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["b"], binding["n"], 3, binding["heads"], binding["dim"]]


def config(binding, dtype):
    b, n, heads, dim = (
        binding["b"],
        binding["n"],
        binding["heads"],
        binding["dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "qkv": {"shape": [b, n, 3, heads, dim], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {
                "shape": [b, n, 3, heads, dim],
                "dtype": dt,
                "note": "q/k 归一化后与 v 一起返回",
            },
        },
        "dims": {"b": b, "n": n, "heads": heads, "dim": dim},
    }
