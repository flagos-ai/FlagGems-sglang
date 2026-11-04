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

"""mla_kv_pack_quantize_fp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8.mla_kv_pack_quantize_fp8
    调用形态复刻 problems/attention/mla_kv_pack_quantize_fp8/baseline.py：
        mla_kv_pack_quantize_fp8(
            k_nope, k_pe, v,
            k_scale_inv=k_scale_inv, v_scale_inv=v_scale_inv, fp8_dtype=fp8_dtype)
    返回新的 fp8 打包张量，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    k_nope = randn(s, heads, qk_nope).bf16；k_pe = randn(s, qk_rope).bf16（pe_3d=False）
    v = randn(s, heads, v_head).bf16；qk_nope=512, qk_rope=64, v_head=512
    k_scale_inv=1.0, v_scale_inv=1.0, fp8_dtype=float8_e4m3fn。seed=0。
dtype 取 cases.py 输入的 [bfloat16]（输出为 fp8）。
"""

import torch

OP_NAME = "mla_kv_pack_quantize_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_QK_NOPE = 512
_QK_ROPE = 64
_V_HEAD = 512
# BENCH_CASES: (s, heads) for s in (1,8,64,512,4096) for heads in (1,16)
_CASES = [(s, h) for s in (1, 8, 64, 512, 4096) for h in (1, 16)]


def native():
    """解析 mla_kv_pack_quantize_fp8.mla_kv_pack_quantize_fp8。"""
    try:
        from sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8 import (
            mla_kv_pack_quantize_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"s": s, "heads": h, "qk_nope": _QK_NOPE, "qk_rope": _QK_ROPE, "v_head": _V_HEAD}
        for (s, h) in _CASES
    ]


def build_inputs(binding, dtype, device):
    s, heads = binding["s"], binding["heads"]
    qk_nope, qk_rope, v_head = binding["qk_nope"], binding["qk_rope"], binding["v_head"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    k_nope = _t(s, heads, qk_nope).contiguous()
    k_pe = _t(s, qk_rope).contiguous()
    v = _t(s, heads, v_head).contiguous()
    return (k_nope, k_pe, v), {
        "k_scale_inv": 1.0,
        "v_scale_inv": 1.0,
        "fp8_dtype": torch.float8_e4m3fn,
    }


def key_shape(binding):
    return "s%d_h%d_nope%d_rope%d_vh%d" % (
        binding["s"], binding["heads"],
        binding["qk_nope"], binding["qk_rope"], binding["v_head"],
    )


def config(binding, dtype):
    s, heads = binding["s"], binding["heads"]
    qk_nope, qk_rope, v_head = binding["qk_nope"], binding["qk_rope"], binding["v_head"]
    dt = str(dtype)
    return {
        "inputs": {
            "k_nope": {"shape": [s, heads, qk_nope], "dtype": dt},
            "k_pe": {"shape": [s, qk_rope], "dtype": dt},
            "v": {"shape": [s, heads, v_head], "dtype": dt},
            "k_scale_inv": {"scalar": 1.0},
            "v_scale_inv": {"scalar": 1.0},
            "fp8_dtype": {"scalar": "torch.float8_e4m3fn"},
        },
        "outputs": {
            "packed": {"shape": [s, heads, qk_nope + qk_rope + v_head],
                       "dtype": "torch.float8_e4m3fn",
                       "note": "k_nope/k_pe/v 打包量化为 fp8"},
        },
        "dims": {
            "s": s, "heads": heads, "qk_nope": qk_nope,
            "qk_rope": qk_rope, "v_head": v_head,
        },
    }
