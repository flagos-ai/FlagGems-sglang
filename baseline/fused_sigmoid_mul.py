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

"""fused_sigmoid_mul baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul
    调用形态复刻 problems/activation_norm/fused_sigmoid_mul/baseline.py：
        fused_sigmoid_mul(attn_output, gate)
    默认 inplace=False -> 另分配输出，不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/activation_norm/fused_sigmoid_mul/cases.py 的 BENCH_CASES：
    flat 分支：attn_output/gate = randn(bs, hidden).to(dtype)
    strided 分支：attn_output = randn(bs, 32*128)，gate = randn(bs, 32, 128)
    （kernel 的 stride(0) 处理路径）。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_sigmoid_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_FLAT = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (2048, 4096, 8192)]
_STRIDED = [(bs, 32, 128) for bs in (1, 64, 1024)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_sigmoid_mul as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    g = [{"kind": "flat", "bs": bs, "hidden": h} for (bs, h) in _FLAT]
    g += [
        {"kind": "strided", "bs": bs, "num_heads": nh, "head_dim": hd}
        for (bs, nh, hd) in _STRIDED
    ]
    return g


def build_inputs(binding, dtype, device):
    bs = binding["bs"]
    g0 = torch.Generator(device=device).manual_seed(0)
    if binding["kind"] == "flat":
        hidden = binding["hidden"]
        attn_output = torch.randn(
            bs, hidden, dtype=torch.float32, device=device, generator=g0
        ).to(dtype)
        g1 = torch.Generator(device=device).manual_seed(1)
        gate = torch.randn(
            bs, hidden, dtype=torch.float32, device=device, generator=g1
        ).to(dtype)
    else:
        nh, hd = binding["num_heads"], binding["head_dim"]
        g10 = torch.Generator(device=device).manual_seed(10)
        attn_output = torch.randn(
            bs, nh * hd, dtype=torch.float32, device=device, generator=g10
        ).to(dtype)
        g11 = torch.Generator(device=device).manual_seed(11)
        gate = torch.randn(
            bs, nh, hd, dtype=torch.float32, device=device, generator=g11
        ).to(dtype)
    return (attn_output, gate), {}


def key_shape(binding):
    if binding["kind"] == "flat":
        return [binding["bs"], binding["hidden"]]
    return "strided_bs{bs}_h{num_heads}_d{head_dim}".format(**binding)


def config(binding, dtype):
    bs = binding["bs"]
    dt = str(dtype)
    if binding["kind"] == "flat":
        hidden = binding["hidden"]
        return {
            "inputs": {
                "attn_output": {"shape": [bs, hidden], "dtype": dt},
                "gate": {"shape": [bs, hidden], "dtype": dt},
            },
            "outputs": {"out": {"shape": [bs, hidden], "dtype": dt}},
            "dims": {"kind": "flat", "bs": bs, "hidden": hidden},
        }
    nh, hd = binding["num_heads"], binding["head_dim"]
    return {
        "inputs": {
            "attn_output": {"shape": [bs, nh * hd], "dtype": dt},
            "gate": {"shape": [bs, nh, hd], "dtype": dt},
        },
        "outputs": {"out": {"shape": [bs, nh * hd], "dtype": dt}},
        "dims": {"kind": "strided", "bs": bs, "num_heads": nh, "head_dim": hd},
    }
