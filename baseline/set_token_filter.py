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

"""Native SGLang collection workload for set_token_filter."""

import torch

OP_NAME = "set_token_filter"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(16, vocab, n) for vocab in (32768,) for n in (16, 1024)]
MORE_SHAPES = [(16, vocab, n) for vocab in (152064,) for n in (16, 1024)]


def native():
    """解析 set_token_filter_triton。"""
    try:
        from sglang.kernels.ops.grammar.token_filter_ops import (
            set_token_filter_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "vocab": vocab,
            "num_tokens": num_tokens,
            **{"is_allowed": True, "reset": True},
        }
        for (batch, vocab, num_tokens) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, vocab, num_tokens = shape
    g = torch.Generator(device=device).manual_seed(9)
    words = (vocab + 31) // 32
    vocab_mask = torch.randint(
        -(2**31),
        2**31 - 1,
        (batch, words),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # token_ids is a Python list, so the permutation stays on the CPU.
    token_ids = torch.randperm(
        vocab, generator=torch.Generator().manual_seed(9)
    )[:num_tokens].tolist()
    return vocab_mask, token_ids, batch // 2, True, True


def build_inputs(binding, dtype, device):
    shape = (binding["batch"], binding["vocab"], binding["num_tokens"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"b{binding['batch']}_V{binding['vocab']}"
        f"_nt{binding['num_tokens']}"
    )


def config(binding, dtype):
    batch, vocab, num_tokens = (
        binding["batch"],
        binding["vocab"],
        binding["num_tokens"],
    )
    is_allowed, reset = binding["is_allowed"], binding["reset"]
    words = (vocab + 31) // 32
    return {
        "inputs": {
            "vocab_mask": {
                "shape": [batch, words],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "token_ids": {
                "note": f"python list，长度 num_tokens={num_tokens}"
            },
            "batch_idx": {"scalar": batch // 2},
            "is_allowed": {"scalar": is_allowed},
            "reset_vocab_mask": {"scalar": reset},
        },
        "outputs": {
            "vocab_mask": {"shape": [batch, words], "dtype": "torch.int32"},
        },
        "dims": {
            "batch": batch,
            "vocab": vocab,
            "words": words,
            "num_tokens": num_tokens,
        },
    }
