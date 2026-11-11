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

"""Native SGLang collection workload for fused_qkvzba_split_reshape_cat."""

import torch

OP_NAME = "fused_qkvzba_split_reshape_cat"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(batch, 16, 32, 128, 128) for batch in (1, 8, 64, 512)]
MORE_SHAPES = [(batch, 16, 32, 128, 128) for batch in (4096,)]


def native():
    """解析 triton_gdn_fused_proj.fused_qkvzba_split_reshape_cat。"""
    try:
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkvzba_split_reshape_cat as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "nqk": nqk,
            "nv": nv,
            "head_qk": head_qk,
            "head_v": head_v,
        }
        for (batch, nqk, nv, head_qk, head_v) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, nqk, nv, head_qk, head_v = shape
    g = torch.Generator(device=device).manual_seed(9)
    r = nv // nqk
    mixed_qkvz = torch.randn(
        batch,
        nqk * (2 * head_qk + 2 * r * head_v),
        device=device,
        generator=g,
    ).to(cur_dtype)
    mixed_ba = torch.randn(batch, nqk * 2 * r, device=device, generator=g).to(
        cur_dtype
    )
    return (
        mixed_qkvz.contiguous(),
        mixed_ba.contiguous(),
        nqk,
        nv,
        head_qk,
        head_v,
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nqk"],
        binding["nv"],
        binding["head_qk"],
        binding["head_v"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return "b%d_nqk%d_nv%d_hqk%d_hv%d" % (
        binding["batch"],
        binding["nqk"],
        binding["nv"],
        binding["head_qk"],
        binding["head_v"],
    )


def config(binding, dtype):
    batch = binding["batch"]
    nqk, nv, head_qk, head_v = (
        binding["nqk"],
        binding["nv"],
        binding["head_qk"],
        binding["head_v"],
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
            "batch": batch,
            "num_heads_qk": nqk,
            "num_heads_v": nv,
            "head_qk": head_qk,
            "head_v": head_v,
            "r": r,
            "qkvz_dim": qkvz_dim,
            "ba_dim": ba_dim,
            "note": "输出为 split/reshape/cat 后的多张量，形状依 kernel 约定",
        },
    }
