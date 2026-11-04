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

"""store_sf_interleaved baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.mxfp8_interleave_sf.store_sf_interleaved
    调用形态复刻 baseline.py：
        out = sf_out.clone(); store_sf_interleaved(sf_in, out, loc, page_size); return out
    kernel 原地写入 out（sf_out，baseline 先 clone）。IS_INPLACE=True：采样前重建实参。

输入构造复刻 cases.py 的 BENCH_CASES（nheads=8, num_pages=256, page_size=128）：
    sf_in=randint(0,256,(t,8,4)) uint8；sf_out=zeros(256,8,32,4,4) uint8；
    loc=randperm(256*128)[:t] int64。
BENCH: _case(t,seed=9) for t in (1,8,64,512,4096,16384)。
"""

import torch

OP_NAME = "store_sf_interleaved"
DTYPES = [torch.uint8]
IS_INPLACE = True  # kernel 原地写入 sf_out

_NHEADS = 8
_NUM_PAGES = 256
_PAGE = 128
_SEED = 9
_SHAPES = [{"t": t} for t in (1, 8, 64, 512, 4096, 16384)]


def native():
    """解析 store_sf_interleaved。"""
    try:
        from sglang.kernels.ops.quantization.mxfp8_interleave_sf import (
            store_sf_interleaved as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    t = binding["t"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    sf_in = torch.randint(
        0, 256, (t, _NHEADS, 4), dtype=torch.uint8, device=device, generator=g
    )
    sf_out = torch.zeros(
        _NUM_PAGES, _NHEADS, 32, 4, 4, dtype=torch.uint8, device=device
    )
    loc = torch.randperm(_NUM_PAGES * _PAGE, device=device, generator=g)[:t].to(
        torch.int64
    )
    return (sf_in, sf_out, loc, _PAGE), {}


def key_shape(binding):
    return f"t{binding['t']}_h{_NHEADS}_np{_NUM_PAGES}_ps{_PAGE}"


def config(binding, dtype):
    t = binding["t"]
    return {
        "inputs": {
            "sf_in": {"shape": [t, _NHEADS, 4], "dtype": "torch.uint8"},
            "sf_out": {"shape": [_NUM_PAGES, _NHEADS, 32, 4, 4],
                       "dtype": "torch.uint8", "note": "原地写回"},
            "loc": {"shape": [t], "dtype": "torch.int64"},
            "page_size": {"scalar": _PAGE},
        },
        "outputs": {
            "sf_out": {"shape": [_NUM_PAGES, _NHEADS, 32, 4, 4], "dtype": "torch.uint8"},
        },
        "dims": {"t": t, "nheads": _NHEADS, "num_pages": _NUM_PAGES,
                 "page_size": _PAGE},
    }
