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

"""normalize_and_patchify baseline（方案 B，sglang）。

native（problem 属 utility 组，符号在 mm.process.image 下）：
    sglang.kernels.ops.mm.process.image.normalize_and_patchify
    调用形态复刻 problems/utility/normalize_and_patchify/baseline.py：
        normalize_and_patchify(image, image_scale, image_bias, patch_size,
                               padded_height, padded_width)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/utility/normalize_and_patchify/cases.py 的 BENCH_CASES：
    image = rand(batch, channels, h, w) fp32; scale = rand(channels,1,1);
    bias = randn(channels,1,1); patch_size=14; padded_height/width=h/w（无 pad）。
    其中 channels=3, h=w=448, patch=14，batch in (1,4,16)。
dtype 取 cases.py BENCH_CASES 的 [float32]。
"""

import torch

OP_NAME = "normalize_and_patchify"
DTYPES = [torch.float32]
IS_INPLACE = False

# BENCH_CASES: batch in (1,4,16); channels=3, h=w=448, patch=14, no pad
_CH = 3
_HW = 448
_PATCH = 14
_BATCH = [1, 4, 16]


def native():
    """解析 sglang.kernels.ops.mm.process.image.normalize_and_patchify。"""
    try:
        from sglang.kernels.ops.mm.process.image import (
            normalize_and_patchify as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "channels": _CH, "h": _HW, "w": _HW, "patch": _PATCH}
        for b in _BATCH
    ]


def build_inputs(binding, dtype, device):
    batch, channels = binding["batch"], binding["channels"]
    h, w, patch = binding["h"], binding["w"], binding["patch"]
    g = torch.Generator(device=device).manual_seed(0)
    image = torch.rand(
        batch, channels, h, w, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    scale = torch.rand(
        channels, 1, 1, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    bias = torch.randn(
        channels, 1, 1, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    return (image, scale, bias, patch, h, w), {}


def key_shape(binding):
    return (f"b{binding['batch']}_c{binding['channels']}"
            f"_h{binding['h']}_w{binding['w']}_p{binding['patch']}")


def config(binding, dtype):
    batch, channels = binding["batch"], binding["channels"]
    h, w, patch = binding["h"], binding["w"], binding["patch"]
    dt = str(dtype)
    num_patches = (h // patch) * (w // patch)
    patch_dim = channels * patch * patch
    return {
        "inputs": {
            "image": {"shape": [batch, channels, h, w], "dtype": dt},
            "image_scale": {"shape": [channels, 1, 1], "dtype": dt},
            "image_bias": {"shape": [channels, 1, 1], "dtype": dt},
            "patch_size": {"scalar": patch},
            "padded_height": {"scalar": h},
            "padded_width": {"scalar": w},
        },
        "outputs": {
            "patches": {"shape": [batch, num_patches, patch_dim], "dtype": dt,
                        "note": "patch 展平输出，形状以 kernel 实际为准"},
        },
        "dims": {
            "batch": batch, "channels": channels, "h": h, "w": w,
            "patch": patch, "num_patches": num_patches, "patch_dim": patch_dim,
        },
    }
