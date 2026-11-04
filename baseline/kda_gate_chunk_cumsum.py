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

"""kda_gate_chunk_cumsum baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.fla.kda.kda_gate_chunk_cumsum
    调用形态复刻 problems/fla/kda_gate_chunk_cumsum/baseline.py：
        kda_gate_chunk_cumsum(g, A_log, chunk_size,
                              scale=scale, dt_bias=dt_bias, lower_bound=lower_bound)
    返回新张量，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    g = randn(b, t, h, s).bf16；A_log = randn(h)*0.5 fp32
    dt_bias=None（bias=False），scale=None，lower_bound=None，chunk_size=64
    BENCH: (1, t, 16, 128) for t in (64,512,2048,8192)。seed=0。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "kda_gate_chunk_cumsum"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_CHUNK_SIZE = 64
# (b, t, h, s)
_CASES = [(1, t, 16, 128) for t in (64, 512, 2048, 8192)]


def native():
    """解析 sglang.kernels.ops.attention.fla.kda.kda_gate_chunk_cumsum。"""
    try:
        from sglang.kernels.ops.attention.fla.kda import kda_gate_chunk_cumsum as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "t": t, "h": h, "s": s, "chunk_size": _CHUNK_SIZE}
        for (b, t, h, s) in _CASES
    ]


def build_inputs(binding, dtype, device):
    b, t, h, s, chunk_size = (
        binding["b"], binding["t"], binding["h"], binding["s"], binding["chunk_size"],
    )
    gen = torch.Generator(device=device).manual_seed(0)
    g = torch.randn(b, t, h, s, dtype=torch.float32, device=device, generator=gen).to(dtype).contiguous()
    a_log = torch.randn(h, dtype=torch.float32, device=device, generator=gen) * 0.5
    return (g, a_log, chunk_size), {"scale": None, "dt_bias": None, "lower_bound": None}


def key_shape(binding):
    return "b%d_t%d_h%d_s%d_cs%d" % (
        binding["b"], binding["t"], binding["h"], binding["s"], binding["chunk_size"],
    )


def config(binding, dtype):
    b, t, h, s, chunk_size = (
        binding["b"], binding["t"], binding["h"], binding["s"], binding["chunk_size"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "g": {"shape": [b, t, h, s], "dtype": dt},
            "A_log": {"shape": [h], "dtype": "torch.float32"},
            "chunk_size": {"scalar": chunk_size},
            "scale": {"scalar": None},
            "dt_bias": {"scalar": None},
            "lower_bound": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [b, t, h, s], "dtype": "torch.float32",
                    "note": "chunk 内累积 gate"},
        },
        "dims": {"b": b, "t": t, "h": h, "s": s, "chunk_size": chunk_size},
    }
