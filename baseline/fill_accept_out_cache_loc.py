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

"""fill_accept_out_cache_loc baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.eagle.fill_accept_out_cache_loc_func
    调用形态复刻 problems/speculative/fill_accept_out_cache_loc/baseline.py：
        out = accept_out_cache_loc.clone()
        fill_accept_out_cache_loc_func(accept_index, out_cache_loc, out, size)
    kernel 原地写第三个实参 out -> IS_INPLACE=True（采集侧每次重建）。

输入构造复刻 cases.py 的 BENCH_CASES：
    accept_index = arange(size) int64，按 rand>0.5 置 -1（拒绝）
    out_cache_loc = randint(1, 1<<20, (size,)) int64
    accept_out_cache_loc = zeros(size) int64
    其中 size in (8,64,512,4096)。
全整型，DTYPES 记 int64。
"""

import torch

OP_NAME = "fill_accept_out_cache_loc"
DTYPES = [torch.int64]
IS_INPLACE = True  # kernel 原地写 accept_out_cache_loc

_ACCEPT_FRAC = 0.5
_SIZES = [8, 64, 512, 4096]


def native():
    """解析 fill_accept_out_cache_loc_func。"""
    try:
        from sglang.kernels.ops.speculative.eagle import (
            fill_accept_out_cache_loc_func as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"size": s, "accept_frac": _ACCEPT_FRAC} for s in _SIZES]


def build_inputs(binding, dtype, device):
    size, accept_frac = binding["size"], binding["accept_frac"]
    g = torch.Generator(device=device).manual_seed(9)
    accept_index = torch.arange(size, dtype=torch.int64, device=device)
    reject = torch.rand(size, device=device, generator=g) > accept_frac
    accept_index[reject] = -1
    out_cache_loc = torch.randint(
        1, 1 << 20, (size,), dtype=torch.int64, device=device, generator=g
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    return (accept_index, out_cache_loc, accept_out_cache_loc, size), {}


def key_shape(binding):
    return [binding["size"]]


def config(binding, dtype):
    size = binding["size"]
    return {
        "inputs": {
            "accept_index": {"shape": [size], "dtype": "torch.int64"},
            "out_cache_loc": {"shape": [size], "dtype": "torch.int64"},
            "accept_out_cache_loc": {"shape": [size], "dtype": "torch.int64",
                                     "note": "原地写回"},
            "size": {"scalar": size},
        },
        "outputs": {
            "accept_out_cache_loc": {"shape": [size], "dtype": "torch.int64"},
        },
        "dims": {"size": size},
    }
