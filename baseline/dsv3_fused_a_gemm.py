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

"""Native SGLang collection workload for dsv3_fused_a_gemm."""

import torch

OP_NAME = "dsv3_fused_a_gemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, 7168, 2112) for m in (1, 2, 4, 8)]
MORE_SHAPES = [(m, 7168, 2112) for m in (16,)]


def native():
    """解析 sglang.kernels.ops.gemm.dsv3_fused_a_gemm.dsv3_fused_a_gemm。"""
    try:
        from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import (
            dsv3_fused_a_gemm as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": num_tokens, "hd_in": hd_in, "hd_out": hd_out}
        for (num_tokens, hd_in, hd_out) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    num_tokens, hd_in, hd_out = (
        binding["num_tokens"],
        binding["hd_in"],
        binding["hd_out"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(
        num_tokens, hd_in, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    # Column-major B: build the row-major [hd_out, hd_in] weight and transpose.
    w = torch.randn(
        hd_out, hd_in, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    return (a, w.t()), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["hd_in"], binding["hd_out"]]


def config(binding, dtype):
    m, hd_in, hd_out = (
        binding["num_tokens"],
        binding["hd_in"],
        binding["hd_out"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "mat_a": {"shape": [m, hd_in], "dtype": dt},
            "mat_b": {
                "shape": [hd_in, hd_out],
                "dtype": dt,
                "note": "列主序（[hd_out, hd_in] 的转置视图）",
            },
        },
        "outputs": {
            "out": {"shape": [m, hd_out], "dtype": dt},
        },
        "dims": {"num_tokens": m, "hd_in": hd_in, "hd_out": hd_out},
    }
