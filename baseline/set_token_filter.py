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

"""set_token_filter baseline（方案 B，sglang）。

native（problem 属 utility 组，符号在 grammar 下）：
    sglang.kernels.ops.grammar.token_filter_ops.set_token_filter_triton
    调用形态复刻 problems/utility/set_token_filter/baseline.py：
        out = vocab_mask.clone()
        set_token_filter_triton(out, token_ids, batch_idx, is_allowed, reset_vocab_mask)
    kernel 原地改写 out（vocab_mask）-> IS_INPLACE=True（采集侧每次重建 mask）。

输入构造复刻 problems/utility/set_token_filter/cases.py 的 BENCH_CASES：
    words = (vocab+31)//32
    vocab_mask = randint(-2^31, 2^31-1, (batch, words)) int32
    token_ids = randperm(vocab)[:num_tokens] 的 python list
    batch_idx = batch//2; is_allowed; reset_vocab_mask
    其中 batch=16，vocab in (32768, 152064)，num_tokens in (16, 1024)。
全整型 mask，DTYPES 记 int32。
"""

import torch

OP_NAME = "set_token_filter"
DTYPES = [torch.int32]
IS_INPLACE = True  # kernel 原地改写 vocab_mask（out）

# BENCH_CASES: batch=16; vocab in (32768,152064) x num_tokens in (16,1024); is_allowed=True reset=True
_BATCH = 16


def native():
    """解析 set_token_filter_triton。"""
    try:
        from sglang.kernels.ops.grammar.token_filter_ops import (
            set_token_filter_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": _BATCH, "vocab": v, "num_tokens": n,
         "is_allowed": True, "reset": True}
        for v in (32768, 152064) for n in (16, 1024)
    ]


def build_inputs(binding, dtype, device):
    batch, vocab, num_tokens = binding["batch"], binding["vocab"], binding["num_tokens"]
    is_allowed, reset = binding["is_allowed"], binding["reset"]
    g = torch.Generator(device=device).manual_seed(9)
    words = (vocab + 31) // 32
    mask = torch.randint(
        -(2 ** 31), 2 ** 31 - 1, (batch, words), dtype=torch.int32,
        device=device, generator=g,
    )
    ids = torch.randperm(vocab, generator=torch.Generator().manual_seed(9))[
        :num_tokens
    ].tolist()
    # baseline 先 out = vocab_mask.clone() 再写 out；采集侧直接传新建的 mask。
    return (mask, ids, batch // 2, is_allowed, reset), {}


def key_shape(binding):
    return (f"b{binding['batch']}_V{binding['vocab']}"
            f"_nt{binding['num_tokens']}")


def config(binding, dtype):
    batch, vocab, num_tokens = binding["batch"], binding["vocab"], binding["num_tokens"]
    is_allowed, reset = binding["is_allowed"], binding["reset"]
    words = (vocab + 31) // 32
    return {
        "inputs": {
            "vocab_mask": {"shape": [batch, words], "dtype": "torch.int32",
                           "note": "原地写回"},
            "token_ids": {"note": f"python list，长度 num_tokens={num_tokens}"},
            "batch_idx": {"scalar": batch // 2},
            "is_allowed": {"scalar": is_allowed},
            "reset_vocab_mask": {"scalar": reset},
        },
        "outputs": {
            "vocab_mask": {"shape": [batch, words], "dtype": "torch.int32"},
        },
        "dims": {
            "batch": batch, "vocab": vocab, "words": words, "num_tokens": num_tokens,
        },
    }
