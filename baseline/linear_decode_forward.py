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

"""linear_decode_forward baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.linear.lightning_attn.linear_decode_forward_triton
    调用形态复刻 problems/attention/linear_decode_forward/baseline.py：
        kv_caches = kv_caches.clone()
        out = linear_decode_forward_triton(
            q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=BLOCK_SIZE)
    kernel 原地更新 kv_caches（故 baseline 先 clone）。IS_INPLACE=True。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    q/k/v = randn(b, h, 1, d) fp32（dtype=fp32）
    kv_caches = randn(num_slots, h, d, d) fp32，num_slots = b + 2
    slope_rate = rand(h) fp32；slot_idx = randperm(num_slots)[:b] int32
    h=8, d=128, BLOCK_SIZE=32。seed=0。
dtype 取 cases.py 的 [float32]。
"""

import torch

OP_NAME = "linear_decode_forward"
DTYPES = [torch.float32]
IS_INPLACE = True  # kernel 原地更新 kv_caches

_H = 8
_D = 128
_BLOCK = 32
_CASES = [1, 8, 64, 256]


def native():
    """解析 linear.lightning_attn.linear_decode_forward_triton。"""
    try:
        from sglang.kernels.ops.attention.linear.lightning_attn import (
            linear_decode_forward_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"b": b, "h": _H, "d": _D, "block": _BLOCK} for b in _CASES]


def build_inputs(binding, dtype, device):
    b, h, d, block = binding["b"], binding["h"], binding["d"], binding["block"]
    num_slots = b + 2
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    q = _t(b, h, 1, d).contiguous()
    k = _t(b, h, 1, d).contiguous()
    v = _t(b, h, 1, d).contiguous()
    kv_caches = _t(num_slots, h, d, d).contiguous()
    slope_rate = torch.rand(h, dtype=torch.float32, device=device, generator=g)
    slot_idx = torch.randperm(num_slots, generator=g, device=device)[:b].to(torch.int32)
    return (q, k, v, kv_caches, slope_rate, slot_idx, block), {}


def key_shape(binding):
    return "b%d_h%d_d%d_blk%d" % (binding["b"], binding["h"], binding["d"], binding["block"])


def config(binding, dtype):
    b, h, d, block = binding["b"], binding["h"], binding["d"], binding["block"]
    num_slots = b + 2
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [b, h, 1, d], "dtype": dt},
            "k": {"shape": [b, h, 1, d], "dtype": dt},
            "v": {"shape": [b, h, 1, d], "dtype": dt},
            "kv_caches": {"shape": [num_slots, h, d, d], "dtype": dt, "note": "原地更新"},
            "slope_rate": {"shape": [h], "dtype": "torch.float32"},
            "slot_idx": {"shape": [b], "dtype": "torch.int32"},
            "BLOCK_SIZE": {"scalar": block},
        },
        "outputs": {
            "out": {"shape": [b, h, d], "dtype": dt},
            "kv_caches": {"shape": [num_slots, h, d, d], "dtype": dt},
        },
        "dims": {"b": b, "h": h, "d": d, "num_slots": num_slots, "block": block},
    }
