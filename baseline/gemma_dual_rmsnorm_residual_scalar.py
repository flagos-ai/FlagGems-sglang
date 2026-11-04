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

"""gemma_dual_rmsnorm_residual_scalar baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.layernorm.gemma4_fused_ops.gemma_dual_rmsnorm_residual_scalar
    调用形态复刻 problems/activation_norm/gemma_dual_rmsnorm_residual_scalar/baseline.py：
        gemma_dual_rmsnorm_residual_scalar(
            x1, weight1, x2, weight2, weight3, residual, scalar,
            eps1=eps1, eps2=eps2, eps3=eps3)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES：
    x1/x2/residual = randn(m, n).contiguous().to(dtype)
    weight1/weight2/weight3 = randn(n).to(dtype)
    scalar = full((1,), 1.25, float32); eps1=eps2=eps3=1e-6。
    同一 generator 顺序取数。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "gemma_dual_rmsnorm_residual_scalar"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (1024, 4096, 8192)]


def native():
    """解析 gemma4_fused_ops.gemma_dual_rmsnorm_residual_scalar。"""
    try:
        from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
            gemma_dual_rmsnorm_residual_scalar as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    x1 = _t(m, n).contiguous()
    weight1 = _t(n)
    x2 = _t(m, n).contiguous()
    weight2 = _t(n)
    weight3 = _t(n)
    residual = _t(m, n).contiguous()
    scalar = torch.full((1,), 1.25, dtype=torch.float32, device=device)
    return (x1, weight1, x2, weight2, weight3, residual, scalar), {
        "eps1": _EPS,
        "eps2": _EPS,
        "eps3": _EPS,
    }


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x1": {"shape": [m, n], "dtype": dt},
            "weight1": {"shape": [n], "dtype": dt},
            "x2": {"shape": [m, n], "dtype": dt},
            "weight2": {"shape": [n], "dtype": dt},
            "weight3": {"shape": [n], "dtype": dt},
            "residual": {"shape": [m, n], "dtype": dt},
            "scalar": {"shape": [1], "dtype": "torch.float32"},
            "eps1": {"scalar": _EPS},
            "eps2": {"scalar": _EPS},
            "eps3": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
