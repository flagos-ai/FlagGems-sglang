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

"""fused_qkvzba_split_reshape_cat baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.triton_gdn_fused_proj.fused_qkvzba_split_reshape_cat
    调用形态复刻 problems/attention/fused_qkvzba_split_reshape_cat/baseline.py：
        fused_qkvzba_split_reshape_cat(
            mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)
    返回新张量，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    r = nv // nqk
    qkvz_dim = nqk * (2*head_qk + 2*r*head_v)；ba_dim = nqk * 2*r
    mixed_qkvz = randn(batch, qkvz_dim).bf16；mixed_ba = randn(batch, ba_dim).bf16
    nqk=16, nv=32, head_qk=128, head_v=128。seed=0。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_qkvzba_split_reshape_cat"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NQK = 16
_NV = 32
_HEAD_QK = 128
_HEAD_V = 128
_CASES = [1, 8, 64, 512, 4096]


def native():
    """解析 triton_gdn_fused_proj.fused_qkvzba_split_reshape_cat。"""
    try:
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkvzba_split_reshape_cat as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "nqk": _NQK, "nv": _NV, "head_qk": _HEAD_QK, "head_v": _HEAD_V}
        for b in _CASES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    nqk, nv, head_qk, head_v = (
        binding["nqk"], binding["nv"], binding["head_qk"], binding["head_v"],
    )
    r = nv // nqk
    qkvz_dim = nqk * (2 * head_qk + 2 * r * head_v)
    ba_dim = nqk * 2 * r
    g = torch.Generator(device=device).manual_seed(0)
    mixed_qkvz = torch.randn(
        batch, qkvz_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    mixed_ba = torch.randn(
        batch, ba_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    return (mixed_qkvz, mixed_ba, nqk, nv, head_qk, head_v), {}


def key_shape(binding):
    return "b%d_nqk%d_nv%d_hqk%d_hv%d" % (
        binding["batch"], binding["nqk"], binding["nv"],
        binding["head_qk"], binding["head_v"],
    )


def config(binding, dtype):
    batch = binding["batch"]
    nqk, nv, head_qk, head_v = (
        binding["nqk"], binding["nv"], binding["head_qk"], binding["head_v"],
    )
    r = nv // nqk
    qkvz_dim = nqk * (2 * head_qk + 2 * r * head_v)
    ba_dim = nqk * 2 * r
    dt = str(dtype)
    return {
        "inputs": {
            "mixed_qkvz": {"shape": [batch, qkvz_dim], "dtype": dt},
            "mixed_ba": {"shape": [batch, ba_dim], "dtype": dt},
            "num_heads_qk": {"scalar": nqk},
            "num_heads_v": {"scalar": nv},
            "head_qk": {"scalar": head_qk},
            "head_v": {"scalar": head_v},
        },
        "outputs": {
            "query": {"shape": [batch, nqk, head_qk], "dtype": dt},
            "key": {"shape": [batch, nqk, head_qk], "dtype": dt},
            "value": {"shape": [batch, nv, head_v], "dtype": dt},
            "z": {"shape": [batch, nv, head_v], "dtype": dt},
            "b": {"shape": [batch, nv], "dtype": dt},
            "a": {"shape": [batch, nv], "dtype": dt},
        },
        "dims": {
            "batch": batch, "num_heads_qk": nqk, "num_heads_v": nv,
            "head_qk": head_qk, "head_v": head_v, "r": r,
            "qkvz_dim": qkvz_dim, "ba_dim": ba_dim,
            "note": "输出为 split/reshape/cat 后的多张量，形状依 kernel 约定",
        },
    }
