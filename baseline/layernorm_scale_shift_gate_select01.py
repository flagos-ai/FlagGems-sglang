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

"""Native SGLang collection workload for layernorm_scale_shift_gate_select01."""

import torch

OP_NAME = "layernorm_scale_shift_gate_select01"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(b, s, 2240) for b in (1, 2) for s in (77, 1024, 4096)]
MORE_SHAPES = [(8, s, 2240) for s in (77, 1024, 4096)]
_EPS = 1e-6


def native():
    """解析 fuse_layernorm_scale_shift_gate_select01_kernel。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.modulate.scale_shift_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.scale_shift import (
            fuse_layernorm_scale_shift_gate_select01_kernel as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.modulate.scale_shift_triton import (
                fuse_layernorm_scale_shift_gate_select01_kernel as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "l": length, "c": c}
        for (b, length, c) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    b, length, c = shape
    g = torch.Generator(device=device).manual_seed(9)

    def _t(*s):
        return torch.randn(
            *s, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    index = torch.randint(
        0, 2, (b, length), device=device, generator=g, dtype=torch.int32
    )
    # weight/bias are None here: the affine path is covered by the test.
    return (
        _t(b, length, c),
        None,
        None,
        _t(b, c),
        _t(b, c),
        _t(b, c),
        _t(b, c),
        _t(b, c),
        _t(b, c),
        index,
        _EPS,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["b"], binding["l"], binding["c"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["b"], binding["l"], binding["c"]]


def config(binding, dtype):
    b, l, c = binding["b"], binding["l"], binding["c"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [b, l, c], "dtype": dt},
            "weight": {"scalar": None},
            "bias": {"scalar": None},
            "scale0": {"shape": [b, c], "dtype": dt},
            "shift0": {"shape": [b, c], "dtype": dt},
            "gate0": {"shape": [b, c], "dtype": dt},
            "scale1": {"shape": [b, c], "dtype": dt},
            "shift1": {"shape": [b, c], "dtype": dt},
            "gate1": {"shape": [b, c], "dtype": dt},
            "index": {"shape": [b, l], "dtype": "torch.int32"},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [b, l, c], "dtype": dt},
        },
        "dims": {"b": b, "l": l, "c": c},
    }
