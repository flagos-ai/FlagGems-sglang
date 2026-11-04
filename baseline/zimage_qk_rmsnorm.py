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

"""zimage_qk_rmsnorm baseline（方案 B，sglang）。

native：复刻 problems/diffusion/zimage_qk_rmsnorm/baseline.py：
    sglang.kernels.ops.diffusion.triton.zimage_native_norm.zimage_qk_rmsnorm_native
        zimage_qk_rmsnorm_native(x, weight, eps)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(b, s, heads, 128).bf16; weight = randn(128).bf16; eps=1e-6;
    head_dim 固定 128（baseline 只接受 128）。BENCH 固定 b=1, heads=32，
    遍历 s∈{128,1024,4360,16384}（from_qkv 缺省 False）。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "zimage_qk_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_HEAD_DIM = 128
_SHAPES = [(1, s, 32) for s in (128, 1024, 4360, 16384)]


def native():
    """解析 zimage_qk_rmsnorm_native。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.zimage_qk_rmsnorm_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.zimage_native_norm import (
            zimage_qk_rmsnorm_native as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.norm.zimage_qk_rmsnorm_triton import (
                zimage_qk_rmsnorm_native as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"b": b, "s": s, "heads": h} for (b, s, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    b, s, heads = binding["b"], binding["s"], binding["heads"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        b, s, heads, _HEAD_DIM, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    weight = torch.randn(_HEAD_DIM, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (x, weight.contiguous(), _EPS), {}


def key_shape(binding):
    return [binding["b"], binding["s"], binding["heads"], _HEAD_DIM]


def config(binding, dtype):
    b, s, heads = binding["b"], binding["s"], binding["heads"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [b, s, heads, _HEAD_DIM], "dtype": dt},
            "weight": {"shape": [_HEAD_DIM], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [b, s, heads, _HEAD_DIM], "dtype": dt},
        },
        "dims": {"b": b, "s": s, "heads": heads, "head_dim": _HEAD_DIM},
    }
