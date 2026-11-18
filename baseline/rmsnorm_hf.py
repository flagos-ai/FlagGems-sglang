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

"""Native SGLang collection workload for rmsnorm_hf."""

import torch

OP_NAME = "rmsnorm_hf"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
_EPS = 1e-6

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (128, 4096, 7168)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (128, 4096, 7168)]


def native():
    """解析 sglang.kernels.ops.layernorm.rmsnorm_hf.rmsnorm_hf。"""
    try:
        from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "hidden": hidden}
        for (bs, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    w = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    return (x, w, _EPS), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [bs, hidden], "dtype": dt},
            "weight": {"shape": [hidden], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
