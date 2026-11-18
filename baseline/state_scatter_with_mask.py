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

"""Native SGLang collection workload for state_scatter_with_mask."""

import torch

OP_NAME = "state_scatter_with_mask"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(8, 512, requests, 4, 16, 128) for requests in (8, 64)]
MORE_SHAPES = [(8, 512, requests, 4, 16, 128) for requests in (256,)]


def native():
    """解析 mamba_state_scatter_triton.fused_mamba_state_scatter_with_mask。"""
    try:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask as op,
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
            "s0": s0,
            "s1": s1,
            **{"state_shape": (s0, s1)},
        }
        for (layers, cache, requests, draft, s0, s1) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    layers, cache, requests, draft, s0, s1 = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    dst = torch.randn(layers, cache, s0, s1, device=device, generator=g).to(
        cur_dtype
    )
    src = torch.randn(
        layers, requests, draft, s0, s1, device=device, generator=g
    ).to(cur_dtype)
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
    return dst.contiguous(), src.contiguous(), dst_idx, step


def build_inputs(binding, dtype, device):
    shape = (
        binding["layers"],
        binding["cache"],
        binding["requests"],
        binding["draft"],
        binding["s0"],
        binding["s1"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    ss = "x".join(str(d) for d in binding["state_shape"])
    return "L%d_c%d_r%d_d%d_ss%s" % (
        binding["layers"],
        binding["cache"],
        binding["requests"],
        binding["draft"],
        ss,
    )


def config(binding, dtype):
    layers, cache, requests = (
        binding["layers"],
        binding["cache"],
        binding["requests"],
    )
    draft, state_shape = binding["draft"], list(binding["state_shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "dst": {
                "shape": [layers, cache] + state_shape,
                "dtype": dt,
                "note": "原地写回",
            },
            "src": {
                "shape": [layers, requests, draft] + state_shape,
                "dtype": dt,
            },
            "dst_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
            "step_indices_raw": {"shape": [requests], "dtype": "torch.int32"},
        },
        "outputs": {
            "dst": {"shape": [layers, cache] + state_shape, "dtype": dt},
        },
        "dims": {
            "layers": layers,
            "cache": cache,
            "requests": requests,
            "draft": draft,
            "state_shape": state_shape,
        },
    }
