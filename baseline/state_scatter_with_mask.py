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

"""state_scatter_with_mask baseline（方案 B，sglang）。

native：sglang.kernels.ops.mamba.mamba_state_scatter_triton.fused_mamba_state_scatter_with_mask
    调用形态复刻 problems/mamba/state_scatter_with_mask/baseline.py：
        dst = dst.clone()
        fused_mamba_state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw)
    kernel 原地 scatter 进 dst（故 baseline 先 clone）。IS_INPLACE=True。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    dst = randn(layers, cache, *state_shape).to(dtype)
    src = randn(layers, requests, draft, *state_shape).to(dtype)
    dst_idx = randperm(cache)[:requests] int32；step = randint(0, draft, (requests,)) int32
    layers=8, cache=512, draft=4, state_shape=(16,128)。seed=0（cases 默认 dtype=fp32）。
    BENCH: requests in (8, 64, 256)。
dtype 取 cases.py BENCH 默认的 [float32]。
"""

import torch

OP_NAME = "state_scatter_with_mask"
DTYPES = [torch.float32]
IS_INPLACE = True  # 原地 scatter 进 dst

_LAYERS = 8
_CACHE = 512
_DRAFT = 4
_STATE_SHAPE = (16, 128)
_CASES = [8, 64, 256]  # requests


def native():
    """解析 mamba_state_scatter_triton.fused_mamba_state_scatter_with_mask。"""
    try:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"layers": _LAYERS, "cache": _CACHE, "requests": r,
         "draft": _DRAFT, "state_shape": list(_STATE_SHAPE)}
        for r in _CASES
    ]


def build_inputs(binding, dtype, device):
    layers, cache, requests = binding["layers"], binding["cache"], binding["requests"]
    draft, state_shape = binding["draft"], list(binding["state_shape"])
    g = torch.Generator(device=device).manual_seed(0)
    dst = torch.randn(
        layers, cache, *state_shape, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    src = torch.randn(
        layers, requests, draft, *state_shape,
        dtype=torch.float32, device=device, generator=g,
    ).to(dtype)
    dst_idx = torch.randperm(cache, generator=g, device=device)[:requests].to(torch.int32)
    step = torch.randint(0, draft, (requests,), device=device, generator=g, dtype=torch.int32)
    dst = dst.contiguous().clone()
    return (dst, src.contiguous(), dst_idx, step), {}


def key_shape(binding):
    ss = "x".join(str(d) for d in binding["state_shape"])
    return "L%d_c%d_r%d_d%d_ss%s" % (
        binding["layers"], binding["cache"], binding["requests"],
        binding["draft"], ss,
    )


def config(binding, dtype):
    layers, cache, requests = binding["layers"], binding["cache"], binding["requests"]
    draft, state_shape = binding["draft"], list(binding["state_shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "dst": {"shape": [layers, cache] + state_shape, "dtype": dt,
                    "note": "原地写回"},
            "src": {"shape": [layers, requests, draft] + state_shape, "dtype": dt},
            "dst_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
            "step_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
        },
        "outputs": {
            "dst": {"shape": [layers, cache] + state_shape, "dtype": dt},
        },
        "dims": {
            "layers": layers, "cache": cache, "requests": requests,
            "draft": draft, "state_shape": state_shape,
        },
    }
