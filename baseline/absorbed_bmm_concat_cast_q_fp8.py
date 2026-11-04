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

"""absorbed_bmm_concat_cast_q_fp8 baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.cache_ops.absorbed_bmm_concat_cast_q_fp8
    调用形态复刻 problems/kvcache/absorbed_bmm_concat_cast_q_fp8/baseline.py：
        q_fp8_pad = q_fp8_pad.clone()
        absorbed_bmm_concat_cast_q_fp8(q_fp8_pad, q_nope, w_kc, q_rope, num_heads)
    kernel 把 absorb-bmm(q_nope, w_kc) 的 nope 结果与 q_rope 拼接、量化写入
    q_fp8_pad -> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES（pad_heads=heads, pad_tokens=tokens）：
    q_fp8_pad = full((tokens, heads, n+rope), 3.0) -> float8_e4m3fn
    q_nope = (0.25*randn(tokens, heads, k)).bf16
    w_kc   = (0.25*randn(heads, k, n)).bf16
    q_rope = (0.25*randn(tokens, heads, rope)).bf16
    num_heads=heads；其中 heads=16, k=128, n=512, rope=64，tokens in (16,128,1024,4096)。
dtype 取 cases.py 的 [bfloat16]（输入张量 dtype；输出 q_fp8_pad 为 fp8）。
"""

import torch

OP_NAME = "absorbed_bmm_concat_cast_q_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 写 q_fp8_pad

_HEADS = 16
_K = 128
_N = 512
_ROPE = 64
_TOKENS = [16, 128, 1024, 4096]


def native():
    """解析 absorbed_bmm_concat_cast_q_fp8。"""
    try:
        from sglang.kernels.ops.kvcache.cache_ops import (
            absorbed_bmm_concat_cast_q_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"tokens": t, "heads": _HEADS, "k": _K, "n": _N, "rope": _ROPE}
        for t in _TOKENS
    ]


def build_inputs(binding, dtype, device):
    tokens, heads = binding["tokens"], binding["heads"]
    k, n, rope = binding["k"], binding["n"], binding["rope"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return (
            0.25 * torch.randn(*shape, dtype=torch.float32, device=device, generator=g)
        ).to(dtype)

    q_fp8_pad = torch.full(
        (tokens, heads, n + rope), 3.0, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    q_nope = _t(tokens, heads, k).contiguous()
    w_kc = _t(heads, k, n).contiguous()
    q_rope = _t(tokens, heads, rope).contiguous()
    return (q_fp8_pad, q_nope, w_kc, q_rope, heads), {}


def key_shape(binding):
    return (f"t{binding['tokens']}_h{binding['heads']}_k{binding['k']}"
            f"_n{binding['n']}_rope{binding['rope']}")


def config(binding, dtype):
    tokens, heads = binding["tokens"], binding["heads"]
    k, n, rope = binding["k"], binding["n"], binding["rope"]
    dt = str(dtype)
    return {
        "inputs": {
            "q_fp8_pad": {"shape": [tokens, heads, n + rope],
                          "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "q_nope": {"shape": [tokens, heads, k], "dtype": dt},
            "w_kc": {"shape": [heads, k, n], "dtype": dt},
            "q_rope": {"shape": [tokens, heads, rope], "dtype": dt},
            "num_heads": {"scalar": heads},
        },
        "outputs": {
            "q_fp8_pad": {"shape": [tokens, heads, n + rope],
                          "dtype": "torch.float8_e4m3fn"},
        },
        "dims": {"tokens": tokens, "heads": heads, "k": k, "n": n, "rope": rope},
    }
