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

"""fill_padded_rows baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.fill_padded_rows._fill_padded_rows（原地）。
复刻 problems/moe/fill_padded_rows/baseline.py：out=x.clone()，调
_fill_padded_rows(out, num_token_non_padded, fill_value)，返回 out。

输入复刻 cases.py BENCH_CASES（rows∈{8,64,512,4096,16384}, cols=8,
valid=rows//2, dtype=int32, fill_value=-1）：x=randint(0,256,(rows,cols)).int32;
num_token_non_padded=tensor([valid]).int32。IS_INPLACE=True（写回 out）。
"""

import torch

OP_NAME = "fill_padded_rows"
DTYPES = [torch.int32]
IS_INPLACE = True

_COLS = 8
_FILL = -1
_ROWS = [8, 64, 512, 4096, 16384]


def native():
    try:
        from sglang.kernels.ops.moe.fill_padded_rows import _fill_padded_rows as _f
    except Exception:
        return None

    def _op(out, num_token_non_padded, fill_value):
        _f(out, num_token_non_padded, fill_value)
        return out

    return _op


def grid():
    return [{"rows": r, "cols": _COLS, "valid": r // 2} for r in _ROWS]


def build_inputs(binding, dtype, device):
    rows, cols, valid = binding["rows"], binding["cols"], binding["valid"]
    g = torch.Generator(device=device).manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(rows, cols, device=device, generator=g).to(dtype)
    else:
        x = torch.randint(0, 256, (rows, cols), dtype=dtype, device=device, generator=g)
    # baseline clone 一次做原地目标。
    out = x.clone()
    num_token_non_padded = torch.tensor([valid], dtype=torch.int32, device=device)
    return (out, num_token_non_padded, _FILL), {}


def key_shape(binding):
    return [binding["rows"], binding["cols"]]


def config(binding, dtype):
    rows, cols, valid = binding["rows"], binding["cols"], binding["valid"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [rows, cols], "dtype": dt, "note": "原地写回（x.clone()）"},
            "num_token_non_padded": {"shape": [1], "dtype": "torch.int32"},
            "fill_value": {"scalar": _FILL},
        },
        "outputs": {
            "x": {"shape": [rows, cols], "dtype": dt},
        },
        "dims": {"rows": rows, "cols": cols, "valid": valid},
    }
