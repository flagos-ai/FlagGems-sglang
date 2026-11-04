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

"""per_tensor_quant_mla_fp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_kernel.per_tensor_quant_mla_fp8
    调用形态复刻 problems/quantization/per_tensor_quant_mla_fp8/baseline.py：
        x_s_out = zeros((1,)) fp32
        per_tensor_quant_mla_fp8(x, x_s_out, eps)
    返回 (q, s)；x_s_out 作为输出 scale 缓冲被写入（每次 build_inputs 新建）。
    IS_INPLACE=True：kernel 写入 x_s_out 实参，采样前重建实参。

输入构造复刻 cases.py 的 BENCH_CASES（head_size=512, eps=1e-12）：
    x=(randn(num_head,num_seq,head_size)*3).to(bf16)。
BENCH: _case(nh,ns,512) for nh in (1,16,128) for ns in (1,32,512,4096)。
"""

import torch

OP_NAME = "per_tensor_quant_mla_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 写入 x_s_out scale 缓冲

_EPS = 1e-12
_HEAD = 512
_SHAPES = [
    {"num_head": nh, "num_seq": ns, "head_size": _HEAD}
    for nh in (1, 16, 128)
    for ns in (1, 32, 512, 4096)
]


def native():
    """解析 per_tensor_quant_mla_fp8。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_tensor_quant_mla_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    nh, ns, hs = binding["num_head"], binding["num_seq"], binding["head_size"]
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        torch.randn(nh, ns, hs, generator=g, device=device, dtype=torch.float32) * 3
    ).to(dtype)
    x_s_out = torch.zeros((1,), dtype=torch.float32, device=device)
    return (x, x_s_out, _EPS), {}


def key_shape(binding):
    return f"nh{binding['num_head']}_ns{binding['num_seq']}_hs{binding['head_size']}"


def config(binding, dtype):
    nh, ns, hs = binding["num_head"], binding["num_seq"], binding["head_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [nh, ns, hs], "dtype": dt},
            "x_s_out": {"shape": [1], "dtype": "torch.float32", "note": "原地写回"},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "q": {"shape": [nh, ns, hs], "dtype": "torch.float8_e4m3fn"},
            "s": {"shape": [1], "dtype": "torch.float32"},
        },
        "dims": {"num_head": nh, "num_seq": ns, "head_size": hs},
    }
