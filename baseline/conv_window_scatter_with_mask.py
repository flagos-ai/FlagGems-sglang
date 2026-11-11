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

"""Native SGLang collection workload for conv_window_scatter_with_mask."""

import torch

OP_NAME = "conv_window_scatter_with_mask"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(8, 512, requests, 4, 128, 4) for requests in (8, 64)]
MORE_SHAPES = [(8, 512, requests, 4, 128, 4) for requests in (256,)]


def native():
    """解析 mamba_state_scatter_triton.fused_conv_window_scatter_with_mask。"""
    try:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "layers": layers,
            "cache": cache,
            "requests": requests,
            "draft": draft,
            "dim": dim,
            "k": k,
        }
        for (layers, cache, requests, draft, dim, k) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    layers, cache, requests, draft, dim, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    win = k - 1
    dst = torch.randn(layers, cache, dim, win, device=device, generator=g).to(
        cur_dtype
    )
    # src is an as_strided window over a shared buffer, so its per-step
    # windows overlap; that layout is the point of this variant.
    base = torch.randn(
        layers, requests, dim, draft + win - 1, device=device, generator=g
    ).to(cur_dtype)
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
    dst_idx = torch.randperm(cache, device="cpu", generator=perm_g)[
        :requests
    ].to(device=device, dtype=torch.int32)
    step = torch.randint(
        0,
        draft,
        (requests,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    return dst.contiguous(), src, dst_idx, step


def build_inputs(binding, dtype, device):
    shape = (
        binding["layers"],
        binding["cache"],
        binding["requests"],
        binding["draft"],
        binding["dim"],
        binding["k"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return "L%d_c%d_r%d_d%d_dim%d_k%d" % (
        binding["layers"],
        binding["cache"],
        binding["requests"],
        binding["draft"],
        binding["dim"],
        binding["k"],
    )


def config(binding, dtype):
    layers, cache, requests = (
        binding["layers"],
        binding["cache"],
        binding["requests"],
    )
    draft, dim, k = binding["draft"], binding["dim"], binding["k"]
    win = k - 1
    dt = str(dtype)
    return {
        "inputs": {
            "dst": {
                "shape": [layers, cache, dim, win],
                "dtype": dt,
                "note": "原地写回",
            },
            "src": {
                "shape": [layers, requests, draft, dim, win],
                "dtype": dt,
                "note": "对共享 buffer 的滑窗 as_strided 非连续视图",
            },
            "dst_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
            "step_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
        },
        "outputs": {
            "dst": {"shape": [layers, cache, dim, win], "dtype": dt},
        },
        "dims": {
            "layers": layers,
            "cache": cache,
            "requests": requests,
            "draft": draft,
            "dim": dim,
            "k": k,
            "win": win,
        },
    }
