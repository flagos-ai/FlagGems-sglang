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

"""memcpy_triton baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.memory.memcpy_triton.memcpy_triton
    调用形态复刻 problems/utility/memcpy_triton/baseline.py：
        out = dst.clone(); memcpy_triton(out, src, dim, offset, sz, offset_src)
    kernel 把 src 的片段写入 dst（out）-> 原地写回第一个实参。
    IS_INPLACE=True（采集侧每次 build_inputs 重建 dst）。

输入构造复刻 problems/utility/memcpy_triton/cases.py 的 BENCH_CASES：
    src = randn(rows, hidden).bf16; dst = zeros(rows, hidden).bf16
    dim=0; offset/sz 为单元素 int32 张量; offset_src=True
    其中 hidden=4096, sz=rows//2, offset=0，rows in (8,64,512,4096,16384)。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "memcpy_triton"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 把 src 片段写入 dst 实参

# BENCH_CASES: rows in (8,64,512,4096,16384); hidden=4096, offset=0, sz=rows//2, offset_src=True
_HIDDEN = 4096
_ROWS = [8, 64, 512, 4096, 16384]


def native():
    """解析 sglang.kernels.ops.memory.memcpy_triton.memcpy_triton。"""
    try:
        from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"rows": r, "hidden": _HIDDEN, "offset": 0, "sz": r // 2,
             "offset_src": True} for r in _ROWS]


def build_inputs(binding, dtype, device):
    rows, hidden = binding["rows"], binding["hidden"]
    offset, sz, offset_src = binding["offset"], binding["sz"], binding["offset_src"]
    g = torch.Generator(device=device).manual_seed(9)
    src = torch.randn(rows, hidden, device=device, generator=g).to(dtype)
    dst = torch.zeros(rows, hidden, dtype=dtype, device=device)
    offset_t = torch.tensor([offset], dtype=torch.int32, device=device)
    sz_t = torch.tensor([sz], dtype=torch.int32, device=device)
    # baseline 先 out = dst.clone() 再写 out；采集侧直接把新建的 dst 当 out 传。
    return (dst, src, 0, offset_t, sz_t, offset_src), {}


def key_shape(binding):
    return (f"rows{binding['rows']}_h{binding['hidden']}"
            f"_sz{binding['sz']}_off{binding['offset']}")


def config(binding, dtype):
    rows, hidden = binding["rows"], binding["hidden"]
    offset, sz, offset_src = binding["offset"], binding["sz"], binding["offset_src"]
    dt = str(dtype)
    return {
        "inputs": {
            "dst": {"shape": [rows, hidden], "dtype": dt, "note": "原地写回"},
            "src": {"shape": [rows, hidden], "dtype": dt},
            "dim": {"scalar": 0},
            "offset": {"shape": [1], "dtype": "torch.int32"},
            "sz": {"shape": [1], "dtype": "torch.int32"},
            "offset_src": {"scalar": offset_src},
        },
        "outputs": {
            "dst": {"shape": [rows, hidden], "dtype": dt},
        },
        "dims": {"rows": rows, "hidden": hidden, "offset": offset, "sz": sz},
    }
