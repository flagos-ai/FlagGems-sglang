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

"""layernorm_scale_shift_gate_select01 baseline（方案 B，sglang）。

native：复刻 problems/diffusion/layernorm_scale_shift_gate_select01/baseline.py：
    sglang.kernels.ops.diffusion.triton.scale_shift
        .fuse_layernorm_scale_shift_gate_select01_kernel(
            x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(b, l, c).bf16; weight/bias=None（BENCH affine 缺省 False）;
    scale0/shift0/gate0/scale1/shift1/gate1 = randn(b, c).bf16;
    index = randint(0, 2, (b, l)) int32; eps=1e-6; BENCH 中 c 固定 2240。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "layernorm_scale_shift_gate_select01"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(b, s, 2240) for b in (1, 2, 8) for s in (77, 1024, 4096)]


def native():
    """解析 fuse_layernorm_scale_shift_gate_select01_kernel。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.modulate.scale_shift_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.scale_shift import (
            fuse_layernorm_scale_shift_gate_select01_kernel as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.modulate.scale_shift_triton import (
                fuse_layernorm_scale_shift_gate_select01_kernel as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"b": b, "l": l, "c": c} for (b, l, c) in _SHAPES]


def build_inputs(binding, dtype, device):
    b, l, c = binding["b"], binding["l"], binding["c"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    x = _t(b, l, c).contiguous()
    scale0 = _t(b, c).contiguous()
    shift0 = _t(b, c).contiguous()
    gate0 = _t(b, c).contiguous()
    scale1 = _t(b, c).contiguous()
    shift1 = _t(b, c).contiguous()
    gate1 = _t(b, c).contiguous()
    index = torch.randint(0, 2, (b, l), device=device, generator=g, dtype=torch.int32)
    # affine 缺省 False -> weight/bias 为 None
    return (x, None, None, scale0, shift0, gate0, scale1, shift1, gate1, index, _EPS), {}


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
