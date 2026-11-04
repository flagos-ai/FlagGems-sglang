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

"""wan_rmsnorm_silu baseline（方案 B，sglang）。

native：复刻 problems/diffusion/wan_rmsnorm_silu/baseline.py：
    sglang.kernels.ops.diffusion.triton.wan_rmsnorm_silu.wan_rmsnorm_silu
        with torch.no_grad(): out = wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps)
    can_use_wan_rmsnorm_silu() 在 grad 开启时会拒绝输入（返回 None），故必须在
    no_grad 下调用。返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(b, c, t, h, w).bf16（channels_last_3d）; gamma = randn(c).bf16;
    bias=None; rms_scale=None; eps=1e-12。
    BENCH 固定 b=1, t=4，遍历 c∈{128,512}, s(h=w)∈{16,32,64}。
dtype 取 cases.py 的 [bfloat16]。

注：native 需在 torch.no_grad() 下调用。采集器直接 op(*args, **kwargs)，故这里用
一个薄包装 _wan_no_grad 固定套上 no_grad，再作为 native() 返回的 callable。
"""

import torch

OP_NAME = "wan_rmsnorm_silu"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-12
_SHAPES = [(1, c, 4, s, s) for c in (128, 512) for s in (16, 32, 64)]


def native():
    """解析 wan_rmsnorm_silu，并包一层 no_grad（can_use_* 在 grad 开启时拒绝输入）。"""
    try:
        from sglang.kernels.ops.diffusion.triton.wan_rmsnorm_silu import (
            wan_rmsnorm_silu as _op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.norm.wan_rmsnorm_silu_triton import (
                wan_rmsnorm_silu as _op,
            )
        except Exception:
            return None
    if not callable(_op):
        return None

    def _wan_no_grad(*args, **kwargs):
        with torch.no_grad():
            return _op(*args, **kwargs)

    return _wan_no_grad


def grid():
    return [
        {"b": b, "c": c, "t": t, "h": h, "w": w} for (b, c, t, h, w) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    b, c, t, h, w = binding["b"], binding["c"], binding["t"], binding["h"], binding["w"]
    g = torch.Generator(device=device).manual_seed(9)
    x = (
        torch.randn(b, c, t, h, w, device=device, generator=g)
        .to(dtype)
        .contiguous(memory_format=torch.channels_last_3d)
    )
    gamma = torch.randn(c, device=device, generator=g).to(dtype)
    return (x, gamma, None, None, _EPS), {}


def key_shape(binding):
    return f"b{binding['b']}_c{binding['c']}_t{binding['t']}_h{binding['h']}_w{binding['w']}"


def config(binding, dtype):
    b, c, t, h, w = binding["b"], binding["c"], binding["t"], binding["h"], binding["w"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [b, c, t, h, w], "dtype": dt, "note": "channels_last_3d"},
            "gamma": {"shape": [c], "dtype": dt},
            "bias": {"scalar": None},
            "rms_scale": {"scalar": None},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [b, c, t, h, w], "dtype": dt},
        },
        "dims": {"b": b, "c": c, "t": t, "h": h, "w": w},
    }
