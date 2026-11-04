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

"""scale_residual_rmsnorm_scale_shift baseline（方案 B，sglang）。

native：复刻 problems/diffusion/scale_residual_rmsnorm_scale_shift/baseline.py：
    sglang.kernels.ops.diffusion.triton.rmsnorm_scale_shift_bitexact
        .fused_scale_residual_rmsnorm_scale_shift_bitexact(
            residual, update, gate, weight, scale, shift, eps)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    residual/update = randn(batch, seq, hidden).bf16;
    gate/scale/shift = randn(batch, 1, hidden).bf16; weight = randn(hidden).bf16;
    eps=1e-6; BENCH 中 hidden 固定 2048。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "scale_residual_rmsnorm_scale_shift"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(b, s, 2048) for b in (1, 2, 8) for s in (77, 1024, 4096)]


def native():
    """解析 fused_scale_residual_rmsnorm_scale_shift_bitexact。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.rmsnorm_scale_shift_bitexact import (
            fused_scale_residual_rmsnorm_scale_shift_bitexact as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact import (
                fused_scale_residual_rmsnorm_scale_shift_bitexact as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"batch": b, "seq": s, "hidden": h} for (b, s, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    batch, seq, hidden = binding["batch"], binding["seq"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    residual = _t(batch, seq, hidden).contiguous()
    update = _t(batch, seq, hidden).contiguous()
    gate = _t(batch, 1, hidden).contiguous()
    weight = _t(hidden).contiguous()
    scale = _t(batch, 1, hidden).contiguous()
    shift = _t(batch, 1, hidden).contiguous()
    return (residual, update, gate, weight, scale, shift, _EPS), {}


def key_shape(binding):
    return [binding["batch"], binding["seq"], binding["hidden"]]


def config(binding, dtype):
    batch, seq, hidden = binding["batch"], binding["seq"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "residual": {"shape": [batch, seq, hidden], "dtype": dt},
            "update": {"shape": [batch, seq, hidden], "dtype": dt},
            "gate": {"shape": [batch, 1, hidden], "dtype": dt},
            "weight": {"shape": [hidden], "dtype": dt},
            "scale": {"shape": [batch, 1, hidden], "dtype": dt},
            "shift": {"shape": [batch, 1, hidden], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [batch, seq, hidden], "dtype": dt,
                    "note": "返回 (normed, new_residual) 中的主输出"},
        },
        "dims": {"batch": batch, "seq": seq, "hidden": hidden},
    }
