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

"""Native SGLang collection workload for gate_topk."""

import torch

OP_NAME = "gate_topk"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, 256, k) for m in (1, 8, 64, 512) for k in (4, 8)]
MORE_SHAPES = [(m, 256, k) for m in (4096,) for k in (4, 8)]


def native():
    try:
        from sglang.kernels.ops.moe.gate_topk import gate_topk as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n, "k": k} for (m, n, k) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    m, n, k = shape
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    g = torch.Generator(device="cpu").manual_seed(0)
    # Distinct values keep the tie-break rule out of the comparison.
    x = (
        torch.randperm(m * n, generator=g).reshape(m, n).float() / (m * n)
    ).to(device)
    return x.contiguous(), {"k": k}


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["n"], binding["k"])
    x, kwargs = build_case(shape, dtype, device)
    return (x, kwargs["k"]), {}


def key_shape(binding):
    return [binding["m"], binding["n"], binding["k"]]


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": "torch.float32"},
            "k": {"scalar": k},
        },
        "outputs": {
            "weights": {"shape": [m, k], "dtype": "torch.float32"},
            "ids": {"shape": [m, k], "dtype": "torch.int32"},
        },
        "dims": {"m": m, "n": n, "k": k},
    }
