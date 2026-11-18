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

"""Native SGLang collection workload for linear_decode_forward."""

import torch

OP_NAME = "linear_decode_forward"
DTYPES = [torch.float32]
IS_INPLACE = True

MUTATED_INPUTS = (3,)

SHAPES = [(b, 8, 128) for b in (1, 8, 64)]
MORE_SHAPES = [(b, 8, 128) for b in (256,)]
_BLOCK_SIZE = 32


def native():
    """解析 linear.lightning_attn.linear_decode_forward_triton。"""
    try:
        from sglang.kernels.ops.attention.linear.lightning_attn import (
            linear_decode_forward_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "h": h, "d": d, **{"block": _BLOCK_SIZE}}
        for (b, h, d) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    b, h, d = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    num_slots = b + 2

    def _t(*size):
        return torch.randn(*size, device=device, generator=g).to(cur_dtype)

    slot_idx = torch.randperm(num_slots, device="cpu", generator=perm_g)[
        :b
    ].to(device=device, dtype=torch.int32)
    return (
        _t(b, h, 1, d).contiguous(),
        _t(b, h, 1, d).contiguous(),
        _t(b, h, 1, d).contiguous(),
        _t(num_slots, h, d, d).contiguous(),
        torch.rand(h, dtype=torch.float32, device=device, generator=g),
        slot_idx,
        _BLOCK_SIZE,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["b"], binding["h"], binding["d"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return "b%d_h%d_d%d_blk%d" % (
        binding["b"],
        binding["h"],
        binding["d"],
        binding["block"],
    )


def config(binding, dtype):
    b, h, d, block = binding["b"], binding["h"], binding["d"], binding["block"]
    num_slots = b + 2
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [b, h, 1, d], "dtype": dt},
            "k": {"shape": [b, h, 1, d], "dtype": dt},
            "v": {"shape": [b, h, 1, d], "dtype": dt},
            "kv_caches": {
                "shape": [num_slots, h, d, d],
                "dtype": dt,
                "note": "原地更新",
            },
            "slope_rate": {"shape": [h], "dtype": "torch.float32"},
            "slot_idx": {"shape": [b], "dtype": "torch.int32"},
            "BLOCK_SIZE": {"scalar": block},
        },
        "outputs": {
            "out": {"shape": [b, h, d], "dtype": dt},
            "kv_caches": {"shape": [num_slots, h, d, d], "dtype": dt},
        },
        "dims": {
            "b": b,
            "h": h,
            "d": d,
            "num_slots": num_slots,
            "block": block,
        },
    }
