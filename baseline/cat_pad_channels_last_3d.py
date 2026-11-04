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

"""cat_pad_channels_last_3d baseline（方案 B，sglang）。

native：复刻 problems/diffusion/cat_pad_channels_last_3d/baseline.py：
    sglang.kernels.ops.diffusion.triton.wan_causal_cache.cat_pad_channels_last_3d
        cat_pad_channels_last_3d(x, cache_x, padding, keep_cache_t)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(b, c, t, h, w).bf16（channels_last_3d）;
    cache_x = randn(b, c, cache_t, h, w).bf16（channels_last_3d）;
    padding=(1,1,1,1,2,0); keep_cache_t=0。
    BENCH 固定 b=1, c=128, h=w=32, cache_t=2，遍历 t∈{4,8,16}。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "cat_pad_channels_last_3d"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_PADDING = (1, 1, 1, 1, 2, 0)
_CACHE_T = 2
_KEEP = 0
_SHAPES = [(1, 128, t, 32, 32) for t in (4, 8, 16)]


def native():
    """解析 cat_pad_channels_last_3d。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.wan_causal_cache_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.wan_causal_cache import (
            cat_pad_channels_last_3d as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.layout.wan_causal_cache_triton import (
                cat_pad_channels_last_3d as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "c": c, "t": t, "h": h, "w": w, "cache_t": _CACHE_T}
        for (b, c, t, h, w) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    b, c, t, h, w = binding["b"], binding["c"], binding["t"], binding["h"], binding["w"]
    cache_t = binding["cache_t"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return (
            torch.randn(*shape, dtype=torch.float32, device=device, generator=g)
            .to(dtype)
            .contiguous(memory_format=torch.channels_last_3d)
        )

    x = _t(b, c, t, h, w)
    cache_x = _t(b, c, cache_t, h, w)
    return (x, cache_x, _PADDING, _KEEP), {}


def key_shape(binding):
    b, c, t = binding["b"], binding["c"], binding["t"]
    h, w, ct = binding["h"], binding["w"], binding["cache_t"]
    return f"b{b}_c{c}_t{t}_h{h}_w{w}_ct{ct}"


def config(binding, dtype):
    b, c, t = binding["b"], binding["c"], binding["t"]
    h, w, cache_t = binding["h"], binding["w"], binding["cache_t"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [b, c, t, h, w], "dtype": dt, "note": "channels_last_3d"},
            "cache_x": {"shape": [b, c, cache_t, h, w], "dtype": dt,
                        "note": "channels_last_3d"},
            "padding": {"scalar": list(_PADDING)},
            "keep_cache_t": {"scalar": _KEEP},
        },
        "outputs": {
            "out": {"note": "拼接 + 填充后的 channels_last_3d 张量（t 维随 padding 变化）"},
        },
        "dims": {
            "b": b, "c": c, "t": t, "h": h, "w": w,
            "cache_t": cache_t, "padding": list(_PADDING),
        },
    }
