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

"""one_pass_rms_norm baseline（方案 B，sglang）。

native：复刻 problems/diffusion/one_pass_rms_norm/baseline.py：
    sglang.kernels.ops.diffusion.triton.rmsnorm_onepass.triton_one_pass_rms_norm
        triton_one_pass_rms_norm(x, w, eps)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/diffusion/one_pass_rms_norm/cases.py 的 BENCH_CASES：
    x = randn(s, d).to(bf16); w = randn(d).to(bf16); eps=1e-6
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "one_pass_rms_norm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(s, d) for s in (1, 64, 1024, 16384) for d in (1536, 3072)]


def native():
    """解析 triton_one_pass_rms_norm。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.rmsnorm_onepass_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.rmsnorm_onepass import (
            triton_one_pass_rms_norm as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.norm.rmsnorm_onepass_triton import (
                triton_one_pass_rms_norm as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"s": s, "d": d} for (s, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    s, d = binding["s"], binding["d"]
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(s, d, device=device, generator=g).to(dtype)
    w = torch.randn(d, device=device, generator=g).to(dtype)
    return (x, w, _EPS), {}


def key_shape(binding):
    return [binding["s"], binding["d"]]


def config(binding, dtype):
    s, d = binding["s"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, d], "dtype": dt},
            "w": {"shape": [d], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [s, d], "dtype": dt},
        },
        "dims": {"s": s, "d": d},
    }
