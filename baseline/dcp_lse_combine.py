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

"""Native SGLang collection workload for dcp_lse_combine."""

import torch

OP_NAME = "dcp_lse_combine"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(n, b, 16, 128) for n in (2, 4, 8) for b in (1, 64)]
MORE_SHAPES = [(n, b, 16, 128) for n in (2, 4, 8) for b in (1024,)]


def native():
    """解析 sglang.kernels.ops.attention.dcp_kernels.dcp_lse_combine_triton。"""
    try:
        from sglang.kernels.ops.attention.dcp_kernels import (
            dcp_lse_combine_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"n": n, "b": b, "h": h, "d": d}
        for (n, b, h, d) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    n, b, h, d = shape
    g = torch.Generator(device=device).manual_seed(9)
    recv_output = torch.randn(n, b, h, d, device=device, generator=g).to(
        cur_dtype
    )
    # The LSE side stays fp32 regardless of the output dtype.
    recv_lse = (
        torch.randn(n, b, h, dtype=torch.float32, device=device, generator=g)
        * 4
    )
    return recv_output.contiguous(), recv_lse.contiguous(), True, True


def build_inputs(binding, dtype, device):
    shape = (binding["n"], binding["b"], binding["h"], binding["d"])
    out, lse, base_on_e, return_lse = build_case(shape, dtype, device)
    return (out, lse), {
        "is_lse_base_on_e": base_on_e,
        "return_lse": return_lse,
    }


def key_shape(binding):
    return [binding["n"], binding["b"], binding["h"], binding["d"]]


def config(binding, dtype):
    n, b, h, d = binding["n"], binding["b"], binding["h"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "recv_output": {"shape": [n, b, h, d], "dtype": dt},
            "recv_lse": {"shape": [n, b, h], "dtype": "torch.float32"},
            "is_lse_base_on_e": {"scalar": True},
            "return_lse": {"scalar": True},
        },
        "outputs": {
            "output": {"shape": [b, h, d], "dtype": dt},
            "lse": {"shape": [b, h], "dtype": "torch.float32"},
        },
        "dims": {"n": n, "b": b, "h": h, "d": d},
    }
