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

"""top_p_renorm baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.sampling.top_p_renorm_triton.top_p_renorm_probs_triton
    调用形态复刻 problems/sampling_grammar/top_p_renorm/baseline.py：
        top_p_renorm_probs_triton(probs, top_p)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/sampling_grammar/top_p_renorm/cases.py 的 BENCH_CASES：
    logits = randn(batch, vocab) fp32; probs = softmax(logits, -1); top_p=0.9
    其中 batch in (1,8,64,256)，vocab in (32000, 131072)。
dtype：probs 为 float32。
"""

import torch

OP_NAME = "top_p_renorm"
DTYPES = [torch.float32]
IS_INPLACE = False

# BENCH_CASES: batch in (1,8,64,256) x vocab in (32000,131072); top_p=0.9
_TOP_P = 0.9
_BATCH = [1, 8, 64, 256]
_VOCAB = [32000, 131072]


def native():
    """解析 top_p_renorm_probs_triton。"""
    try:
        from sglang.kernels.ops.sampling.top_p_renorm_triton import (
            top_p_renorm_probs_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"batch": b, "vocab": v} for b in _BATCH for v in _VOCAB]


def build_inputs(binding, dtype, device):
    batch, vocab = binding["batch"], binding["vocab"]
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(batch, vocab, dtype=torch.float32, device=device, generator=g)
    probs = torch.softmax(logits, dim=-1).contiguous()
    return (probs, _TOP_P), {}


def key_shape(binding):
    return [binding["batch"], binding["vocab"]]


def config(binding, dtype):
    batch, vocab = binding["batch"], binding["vocab"]
    return {
        "inputs": {
            "probs": {"shape": [batch, vocab], "dtype": "torch.float32"},
            "top_p": {"scalar": _TOP_P},
        },
        "outputs": {
            "out": {"shape": [batch, vocab], "dtype": "torch.float32"},
        },
        "dims": {"batch": batch, "vocab": vocab},
    }
