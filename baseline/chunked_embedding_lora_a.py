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

"""chunked_embedding_lora_a baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.gemm.chunked_embedding_lora_a.chunked_embedding_lora_a_forward
    调用形态复刻 problems/lora/chunked_embedding_lora_a/baseline.py：
        chunked_embedding_lora_a_forward(input_ids, weights, batch_info, vocab_size)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/lora/chunked_embedding_lora_a/cases.py 的 BENCH_CASES
（num_extra=0，故无 extra_embeddings）：
    total_vocab = vocab_size + num_extra = vocab_size
    input_ids = randint(0, total_vocab, (s,), int64)
    weights = randn(num_lora, r, vocab_size).to(dtype)
    batch_info permutation="none"（_case 未传 permutation，make_batch_info 默认）。
dtype 取 benchmark/test_chunked_embedding_lora_a.py 的 [bfloat16]。
"""

import torch

from _lora_batch_utils import make_batch_info

OP_NAME = "chunked_embedding_lora_a"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# BENCH_CASES: (seg_lens, num_lora, r, vocab_size)  —— num_extra 恒为 0
_CASES = [
    ([512] * 8, 4, 32, 32000),
    ([2048] * 4, 2, 64, 128256),
]


def native():
    """解析 chunked_embedding_lora_a.chunked_embedding_lora_a_forward。"""
    try:
        from sglang.kernels.ops.gemm.chunked_embedding_lora_a import (
            chunked_embedding_lora_a_forward as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"seg_lens": list(sl), "num_lora": nl, "r": r, "vocab_size": v}
        for (sl, nl, r, v) in _CASES
    ]


def build_inputs(binding, dtype, device):
    seg_lens = binding["seg_lens"]
    num_lora, r, vocab_size = binding["num_lora"], binding["r"], binding["vocab_size"]
    s = sum(seg_lens)
    total_vocab = vocab_size  # num_extra == 0
    g = torch.Generator(device=device).manual_seed(0)
    input_ids = torch.randint(
        0, total_vocab, (s,), generator=g, device=device, dtype=torch.int64
    )
    weights = torch.randn(
        num_lora, r, vocab_size, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    batch_info = make_batch_info(
        seg_lens, weight_indices, [r] * num_lora, device, permutation="none"
    )
    return (input_ids, weights, batch_info, vocab_size), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_V{binding['vocab_size']}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, vocab_size = (
        binding["num_lora"], binding["r"], binding["vocab_size"],
    )
    s = sum(seg_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "input_ids": {"shape": [s], "dtype": "torch.int64"},
            "weights": {"shape": [num_lora, r, vocab_size], "dtype": dt},
            "batch_info": {"note": "LoRABatchInfo（seg_indptr/weight_indices/"
                           "lora_ranks/scalings 等 LoRA 分段路由 metadata）"},
            "vocab_size": {"scalar": vocab_size},
        },
        "outputs": {
            "output": {"shape": [s, r], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens), "s": s,
            "num_lora": num_lora, "r": r, "vocab_size": vocab_size,
        },
    }
