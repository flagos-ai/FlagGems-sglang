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

"""Native SGLang collection workload for softcap_inplace_logits."""

import torch

OP_NAME = "softcap_inplace_logits"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SOFTCAP_CONST = 30.0
SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (4096, 32000, 128256)]
MORE_SHAPES = [(1, 17), (37, 1024), (4, 32000)]


def native():
    """解析 sglang.kernels.ops.activation.softcap.softcap_inplace_logits。"""
    try:
        from sglang.kernels.ops.activation.softcap import (
            softcap_inplace_logits as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in (*SHAPES, *MORE_SHAPES)]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    full_logits = (
        torch.randn(m, n, generator=g, device=device, dtype=torch.float32) * 20
    ).to(dtype)
    return (full_logits, SOFTCAP_CONST), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "full_logits": {"shape": [m, n], "dtype": dt},
            "final_logit_softcapping": {"scalar": SOFTCAP_CONST},
        },
        "outputs": {
            "full_logits": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
