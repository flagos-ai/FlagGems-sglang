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

"""gather_dequant_requant_fp8_paged baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsa.dequant_k_cache.
        gather_dequant_requant_fp8_paged
    调用形态复刻 baseline.py：
        out = out.clone()
        gather_dequant_requant_fp8_paged(quant_k_cache, page_table_1_flattened,
            group_size=group_size, extra_rows=extra_rows, out=out)
    kernel 全量覆写 out（baseline 先 clone）。IS_INPLACE=True：采样前重建实参。

输入构造复刻 cases.py 的 BENCH_CASES（dim=656, group=128, total_cached=65536,
extra_rows=128）：打包的 656 字节行 = fp8 nope(512) + fp32 group scale(16) + bf16 rope(128)；
page_table=randint int32 (num_tokens,)；out=full((num_tokens+extra,1,576),3.0).to(fp8)。
BENCH: _case(65536, t, 128) for t in (16,128,1024,4096,16384)。
"""

import torch

OP_NAME = "gather_dequant_requant_fp8_paged"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 全量覆写 out

_DIM = 656
_GROUP = 128
_OUT_DIM = 576
_FP8 = torch.float8_e4m3fn
_SHAPES = [
    {"total_cached": 65536, "num_tokens": t, "extra_rows": 128}
    for t in (16, 128, 1024, 4096, 16384)
]


def native():
    """解析 gather_dequant_requant_fp8_paged。"""
    try:
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            gather_dequant_requant_fp8_paged as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    total_cached, num_tokens, extra_rows = (
        binding["total_cached"], binding["num_tokens"], binding["extra_rows"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    cache = torch.empty((total_cached, 1, _DIM), dtype=_FP8, device=device)
    flat = cache.view(total_cached, _DIM)
    flat[:, :512] = (
        torch.randn(total_cached, 512, dtype=torch.float32, device=device, generator=g)
    ).to(_FP8)
    scales = 0.05 * (
        1.0 + torch.rand(total_cached, 4, dtype=torch.float32, device=device, generator=g)
    )
    flat[:, 512:528] = scales.view(_FP8)
    rope = torch.randn(
        total_cached, 64, dtype=torch.float32, device=device, generator=g
    ).to(torch.bfloat16)
    flat[:, 528:] = rope.view(_FP8)
    page_table = torch.randint(
        0, total_cached, (num_tokens,), device=device, generator=g, dtype=torch.int32
    )
    out = torch.full(
        (num_tokens + extra_rows, 1, _OUT_DIM), 3.0, dtype=torch.float32, device=device
    ).to(_FP8)
    return (cache, page_table, _GROUP, extra_rows, out), {}


def key_shape(binding):
    return (f"tc{binding['total_cached']}_t{binding['num_tokens']}"
            f"_ex{binding['extra_rows']}")


def config(binding, dtype):
    total_cached, num_tokens, extra_rows = (
        binding["total_cached"], binding["num_tokens"], binding["extra_rows"],
    )
    return {
        "inputs": {
            "quant_k_cache": {"shape": [total_cached, 1, _DIM],
                              "dtype": "torch.float8_e4m3fn",
                              "note": "打包 fp8 nope + fp32 scale + bf16 rope"},
            "page_table_1_flattened": {"shape": [num_tokens], "dtype": "torch.int32"},
            "group_size": {"scalar": _GROUP},
            "extra_rows": {"scalar": extra_rows},
            "out": {"shape": [num_tokens + extra_rows, 1, _OUT_DIM],
                    "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
        },
        "outputs": {
            "out": {"shape": [num_tokens + extra_rows, 1, _OUT_DIM],
                    "dtype": "torch.float8_e4m3fn"},
        },
        "dims": {"total_cached": total_cached, "num_tokens": num_tokens,
                 "extra_rows": extra_rows, "dim": _DIM, "out_dim": _OUT_DIM,
                 "group_size": _GROUP},
    }
