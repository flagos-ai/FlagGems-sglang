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

"""Native SGLang collection workload for log_scaling_tau."""

import torch

OP_NAME = "log_scaling_tau"
DTYPES = [torch.float16]
IS_INPLACE = False

SHAPES = [(t, 16, 128) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 1, 128), (37, 8, 64), (129, 1, 512)]


def native():
    """解析 sglang.kernels.ops.attention.log_scaling_tau.apply_log_scaling_tau。"""
    try:
        from sglang.kernels.ops.attention.log_scaling_tau import (
            apply_log_scaling_tau as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"t": t, "heads": heads, "d": d}
        for (t, heads, d) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    t, heads, d = binding["t"], binding["heads"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    dims = (t, d) if heads == 1 else (t, heads, d)
    x = torch.randn(*dims, generator=g, device=device, dtype=torch.float32).to(
        dtype
    )
    # tau is per-row (one entry per index along dim 0).
    tau = 0.5 + torch.rand(t, generator=g, device=device, dtype=torch.float32)
    return (x, tau), {}


def key_shape(binding):
    return [binding["t"], binding["heads"], binding["d"]]


def config(binding, dtype):
    t, heads, d = binding["t"], binding["heads"], binding["d"]
    dt = str(dtype)
    shape = [t, d] if heads == 1 else [t, heads, d]
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": dt},
            "tau": {"shape": [t], "dtype": "torch.float32", "note": "per-row"},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": dt},
        },
        "dims": {"t": t, "heads": heads, "d": d},
    }
