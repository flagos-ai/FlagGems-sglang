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

"""conv_window_scatter_with_mask baseline（方案 B，sglang）。

native：sglang.kernels.ops.mamba.mamba_state_scatter_triton.fused_conv_window_scatter_with_mask
    调用形态复刻 problems/mamba/conv_window_scatter_with_mask/baseline.py：
        dst = dst.clone()
        fused_conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw)
    kernel 原地 scatter 进 dst（故 baseline 先 clone）。IS_INPLACE=True。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    win = k - 1
    dst = randn(layers, cache, dim, win).to(dtype)
    base = randn(layers, requests, dim, draft+win-1).to(dtype)
    src = base.as_strided((layers, requests, draft, dim, win), 滑窗 stride)  # 非连续视图
    dst_idx = randperm(cache)[:requests] int32；step = randint(0, draft, (requests,)) int32
    layers=8, draft=4, dim=128, k=4, invalid=0.0。seed=0（cases 默认 dtype=fp32）。
    BENCH: requests in (8, 64, 256)。
dtype 取 cases.py BENCH 默认的 [float32]。
"""

import torch

OP_NAME = "conv_window_scatter_with_mask"
DTYPES = [torch.float32]
IS_INPLACE = True  # 原地 scatter 进 dst

_LAYERS = 8
_DRAFT = 4
_DIM = 128
_K = 4
_CASES = [8, 64, 256]  # requests


def native():
    """解析 mamba_state_scatter_triton.fused_conv_window_scatter_with_mask。"""
    try:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"layers": _LAYERS, "cache": 512, "requests": r,
         "draft": _DRAFT, "dim": _DIM, "k": _K}
        for r in _CASES
    ]


def build_inputs(binding, dtype, device):
    layers, cache, requests = binding["layers"], binding["cache"], binding["requests"]
    draft, dim, k = binding["draft"], binding["dim"], binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    win = k - 1
    dst = torch.randn(
        layers, cache, dim, win, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    base = torch.randn(
        layers, requests, dim, draft + win - 1,
        dtype=torch.float32, device=device, generator=g,
    ).to(dtype)
    src = base.as_strided(
        (layers, requests, draft, dim, win),
        (
            base.stride(0),
            base.stride(1),
            base.stride(3),
            base.stride(2),
            base.stride(3),
        ),
    )
    dst_idx = torch.randperm(cache, generator=g, device=device)[:requests].to(torch.int32)
    step = torch.randint(0, draft, (requests,), device=device, generator=g, dtype=torch.int32)
    dst = dst.contiguous().clone()
    return (dst, src, dst_idx, step), {}


def key_shape(binding):
    return "L%d_c%d_r%d_d%d_dim%d_k%d" % (
        binding["layers"], binding["cache"], binding["requests"],
        binding["draft"], binding["dim"], binding["k"],
    )


def config(binding, dtype):
    layers, cache, requests = binding["layers"], binding["cache"], binding["requests"]
    draft, dim, k = binding["draft"], binding["dim"], binding["k"]
    win = k - 1
    dt = str(dtype)
    return {
        "inputs": {
            "dst": {"shape": [layers, cache, dim, win], "dtype": dt, "note": "原地写回"},
            "src": {"shape": [layers, requests, draft, dim, win], "dtype": dt,
                    "note": "对共享 buffer 的滑窗 as_strided 非连续视图"},
            "dst_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
            "step_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
        },
        "outputs": {
            "dst": {"shape": [layers, cache, dim, win], "dtype": dt},
        },
        "dims": {
            "layers": layers, "cache": cache, "requests": requests,
            "draft": draft, "dim": dim, "k": k, "win": win,
        },
    }
