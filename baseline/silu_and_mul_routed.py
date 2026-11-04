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

"""silu_and_mul_routed baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.inkling_moe.silu_and_mul。复刻 problems/moe/
silu_and_mul_routed/baseline.py：silu_and_mul(gateup_output, topk_weights,
out_dtype=out_dtype, use_interleaved=use_interleaved)，返回新张量 ->
IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（m∈{1,8,64,512,4096}, n∈{1024,4096}, weights=True,
interleaved=True, out_dtype=None）：gateup=randn(m,2n).bf16; topk_weights=rand(m).bf16。
"""

import torch

OP_NAME = "silu_and_mul_routed"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (1024, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.inkling_moe import silu_and_mul as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    gateup = torch.randn(m, 2 * n, dtype=torch.float32, device=device, generator=g).to(dtype).contiguous()
    w = torch.rand(m, dtype=torch.float32, device=device, generator=g).to(dtype)
    return (gateup, w), {"out_dtype": None, "use_interleaved": True}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "gateup_output": {"shape": [m, 2 * n], "dtype": dt},
            "topk_weights": {"shape": [m], "dtype": dt},
            "out_dtype": {"scalar": None},
            "use_interleaved": {"scalar": True},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n, "gateup_last": 2 * n},
    }
