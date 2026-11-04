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

"""reshape_and_cache_flash baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.cache_ops.launch_reshape_and_cache_flash
    调用形态复刻 problems/kvcache/reshape_and_cache_flash/baseline.py：
        key_cache = key_cache.clone(); value_cache = value_cache.clone()
        launch_reshape_and_cache_flash(key, value, key_cache, value_cache,
            slot_mapping, swa_slot_mapping=..., k_scale=..., v_scale=...)
    kernel 把 key/value 按 slot_mapping 写入 key_cache/value_cache -> 原地写回。
    IS_INPLACE=True（采集侧每次重建 cache）。

输入构造复刻 cases.py 的 BENCH_CASES：
    key/value = randn(tokens, heads, head_size).bf16
    key_cache/value_cache = randn(blocks, block_size, heads, head_size).bf16
    slot_mapping = randperm(blocks*block_size)[:tokens] int64
    swa_slot_mapping=None, k_scale=None, v_scale=None
    其中 heads=8, head_size=128, block_size=16, blocks=max(4, t//16+1)，
    tokens(t) in (16,128,1024,4096)。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "reshape_and_cache_flash"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 把 key/value 写入 key_cache/value_cache

_HEADS = 8
_HEAD_SIZE = 128
_BLOCK_SIZE = 16
_TOKENS = [16, 128, 1024, 4096]


def native():
    """解析 launch_reshape_and_cache_flash。"""
    try:
        from sglang.kernels.ops.kvcache.cache_ops import (
            launch_reshape_and_cache_flash as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"tokens": t, "heads": _HEADS, "head_size": _HEAD_SIZE,
         "block_size": _BLOCK_SIZE, "blocks": max(4, t // 16 + 1)}
        for t in _TOKENS
    ]


def build_inputs(binding, dtype, device):
    tokens, heads, head_size = binding["tokens"], binding["heads"], binding["head_size"]
    block_size, blocks = binding["block_size"], binding["blocks"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    total_slots = blocks * block_size
    slots = torch.randperm(total_slots, generator=g, device=device)[:tokens].to(torch.int64)
    key = _t(tokens, heads, head_size).contiguous()
    value = _t(tokens, heads, head_size).contiguous()
    key_cache = _t(blocks, block_size, heads, head_size).contiguous()
    value_cache = _t(blocks, block_size, heads, head_size).contiguous()
    return (key, value, key_cache, value_cache, slots), {
        "swa_slot_mapping": None, "k_scale": None, "v_scale": None,
    }


def key_shape(binding):
    return (f"t{binding['tokens']}_h{binding['heads']}_hd{binding['head_size']}"
            f"_bs{binding['block_size']}_blk{binding['blocks']}")


def config(binding, dtype):
    tokens, heads, head_size = binding["tokens"], binding["heads"], binding["head_size"]
    block_size, blocks = binding["block_size"], binding["blocks"]
    dt = str(dtype)
    return {
        "inputs": {
            "key": {"shape": [tokens, heads, head_size], "dtype": dt},
            "value": {"shape": [tokens, heads, head_size], "dtype": dt},
            "key_cache": {"shape": [blocks, block_size, heads, head_size],
                          "dtype": dt, "note": "原地写回"},
            "value_cache": {"shape": [blocks, block_size, heads, head_size],
                            "dtype": dt, "note": "原地写回"},
            "slot_mapping": {"shape": [tokens], "dtype": "torch.int64"},
            "swa_slot_mapping": {"scalar": None},
            "k_scale": {"scalar": None},
            "v_scale": {"scalar": None},
        },
        "outputs": {
            "key_cache": {"shape": [blocks, block_size, heads, head_size], "dtype": dt},
            "value_cache": {"shape": [blocks, block_size, heads, head_size], "dtype": dt},
        },
        "dims": {
            "tokens": tokens, "heads": heads, "head_size": head_size,
            "block_size": block_size, "blocks": blocks,
        },
    }
