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

"""Native SGLang collection workload for rmsnorm_scale_shift."""

import torch

OP_NAME = "rmsnorm_scale_shift"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(b, s, 2048) for b in (1, 2) for s in (77, 1024, 4096)]
MORE_SHAPES = [(8, s, 2048) for s in (77, 1024, 4096)]
EPS = 1e-6


def native():
    """解析 fused_rmsnorm_scale_shift_bitexact。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.rmsnorm_scale_shift_bitexact import (
            fused_rmsnorm_scale_shift_bitexact as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact import (
                fused_rmsnorm_scale_shift_bitexact as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": batch, "seq": seq, "hidden": hidden}
        for (batch, seq, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    batch, seq, hidden = binding["batch"], binding["seq"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    return (
        _t(batch, seq, hidden),
        _t(hidden),
        _t(batch, 1, hidden),
        _t(batch, 1, hidden),
        EPS,
    ), {}


def key_shape(binding):
    return [binding["batch"], binding["seq"], binding["hidden"]]


def config(binding, dtype):
    batch, seq, hidden = binding["batch"], binding["seq"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [batch, seq, hidden], "dtype": dt},
            "weight": {"shape": [hidden], "dtype": dt},
            "scale": {"shape": [batch, 1, hidden], "dtype": dt},
            "shift": {"shape": [batch, 1, hidden], "dtype": dt},
            "eps": {"scalar": EPS},
        },
        "outputs": {
            "out": {"shape": [batch, seq, hidden], "dtype": dt},
        },
        "dims": {"batch": batch, "seq": seq, "hidden": hidden},
    }
