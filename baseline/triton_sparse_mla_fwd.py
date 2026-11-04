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

"""triton_sparse_mla_fwd baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd
    调用形态复刻 baseline.py：
        triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v=d_v)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（dim=576, d_v=512, h=16, num_pages=8192）：
    q_nope=(0.5*randn(seq,h,512)).to(fp8)；q_rope=(0.5*randn(seq,h,64)).to(fp8)；
    kv=(0.5*randn(num_pages,1,576)).to(fp8)；indices=randint(0,num_pages,(seq,1,topk)) int32；
    sm_scale=576**-0.5。
BENCH: _case(seq,16,topk,num_pages=8192) for seq in (1,16,128) for topk in (128,512,2048)。
"""

import torch

OP_NAME = "triton_sparse_mla_fwd"
DTYPES = [torch.float8_e4m3fn]
IS_INPLACE = False

_DIM = 576
_D_V = 512
_H = 16
_NUM_PAGES = 8192
_CASES = [
    (seq, _H, topk, _NUM_PAGES)
    for seq in (1, 16, 128)
    for topk in (128, 512, 2048)
]


def native():
    """解析 triton_sparse_mla_fwd。"""
    try:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            triton_sparse_mla_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seq": seq, "h": h, "topk": topk, "num_pages": np}
        for (seq, h, topk, np) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seq, h, topk, num_pages = (
        binding["seq"], binding["h"], binding["topk"], binding["num_pages"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    fp8 = torch.float8_e4m3fn

    def _q(width):
        return (
            0.5 * torch.randn(seq, h, width, dtype=torch.float32, device=device, generator=g)
        ).to(fp8)

    kv = (
        0.5 * torch.randn(num_pages, 1, _DIM, dtype=torch.float32, device=device, generator=g)
    ).to(fp8)
    idx = torch.randint(
        0, num_pages, (seq, 1, topk), device=device, generator=g, dtype=torch.int32
    )
    return (_q(_D_V), _q(_DIM - _D_V), kv, idx, _DIM ** -0.5), {"d_v": _D_V}


def key_shape(binding):
    return (f"seq{binding['seq']}_h{binding['h']}_topk{binding['topk']}"
            f"_np{binding['num_pages']}")


def config(binding, dtype):
    seq, h, topk, num_pages = (
        binding["seq"], binding["h"], binding["topk"], binding["num_pages"],
    )
    return {
        "inputs": {
            "q_nope": {"shape": [seq, h, _D_V], "dtype": "torch.float8_e4m3fn"},
            "q_rope": {"shape": [seq, h, _DIM - _D_V], "dtype": "torch.float8_e4m3fn"},
            "kv": {"shape": [num_pages, 1, _DIM], "dtype": "torch.float8_e4m3fn"},
            "indices": {"shape": [seq, 1, topk], "dtype": "torch.int32"},
            "sm_scale": {"scalar": _DIM ** -0.5},
            "d_v": {"scalar": _D_V},
        },
        "outputs": {
            "out": {"shape": [seq, h, _D_V], "dtype": "torch.bfloat16"},
        },
        "dims": {"seq": seq, "h": h, "topk": topk, "num_pages": num_pages,
                 "dim": _DIM, "d_v": _D_V},
    }
