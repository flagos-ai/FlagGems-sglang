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

"""indexed_scale_shift baseline（方案 B，sglang）。

native：复刻 problems/diffusion/indexed_scale_shift/baseline.py：
    sglang.kernels.ops.diffusion.triton.indexed_modulation.indexed_scale_shift_bf16_
        out = x.clone(); indexed_scale_shift_bf16_(out, shift, scale, indices)
    kernel 原地修改第一个实参 out -> IS_INPLACE=True。采集侧每次 build_inputs
    新建 out（等价 baseline 的 clone），对外部无副作用；NCU 采样前重建避免 warmup 污染。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(rows, hidden).bf16; shift/scale = randn(variants, hidden).bf16;
    indices = randint(0, variants, (rows,)) int32; variants=8; hidden 固定 3072。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "indexed_scale_shift"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地修改 out 实参

_VARIANTS = 8
_SHAPES = [(rows, 3072) for rows in (1, 64, 1024, 8192, 32768)]


def native():
    """解析 indexed_scale_shift_bf16_。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
            indexed_scale_shift_bf16_ as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
                indexed_scale_shift_bf16_ as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"rows": r, "hidden": h, "variants": _VARIANTS} for (r, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    rows, hidden, variants = binding["rows"], binding["hidden"], binding["variants"]
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(rows, hidden, device=device, generator=g).to(dtype)
    shift = torch.randn(variants, hidden, device=device, generator=g).to(dtype)
    scale = torch.randn(variants, hidden, device=device, generator=g).to(dtype)
    indices = torch.randint(
        0, variants, (rows,), dtype=torch.int32, device=device, generator=g
    )
    # baseline 先 clone 再原地改；采集侧直接新建 out 作为被写回的实参。
    out = x.clone()
    return (out, shift, scale, indices), {}


def key_shape(binding):
    return [binding["rows"], binding["hidden"]]


def config(binding, dtype):
    rows, hidden, variants = binding["rows"], binding["hidden"], binding["variants"]
    dt = str(dtype)
    return {
        "inputs": {
            "out": {"shape": [rows, hidden], "dtype": dt, "note": "原地写回（x.clone()）"},
            "shift": {"shape": [variants, hidden], "dtype": dt},
            "scale": {"shape": [variants, hidden], "dtype": dt},
            "indices": {"shape": [rows], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [rows, hidden], "dtype": dt},
        },
        "dims": {"rows": rows, "hidden": hidden, "variants": variants},
    }
