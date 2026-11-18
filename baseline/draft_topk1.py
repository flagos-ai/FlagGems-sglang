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

"""Native SGLang collection workload for draft_topk1."""

import torch

OP_NAME = "draft_topk1"
DTYPES = [torch.float32]
IS_INPLACE = True

NUM_DRAFT_COLS = 4
SHAPES = [(bs, 151936, True, 0) for bs in (1, 8, 64, 512)]
MORE_SHAPES = [
    (1, 32000, False, 0),
    (37, 12000, True, 2),
    (129, 8192 + 500, False, 0),
    (8, 151936, True, 0),
]


def native():
    """解析 draft_topk1_postprocess。"""
    try:
        from sglang.kernels.ops.speculative.topk1 import (
            draft_topk1_postprocess as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": bs,
            "vocab_size": vocab_size,
            "with_draft_tokens": with_draft_tokens,
            "draft_token_column": draft_token_column,
        }
        for (
            bs,
            vocab_size,
            with_draft_tokens,
            draft_token_column,
        ) in dict.fromkeys((*SHAPES, *MORE_SHAPES))
    ]


def build_case(shape, cur_dtype, device):
    bs, vocab_size, with_draft_tokens, draft_token_column = shape
    g = torch.Generator(device=device).manual_seed(0)
    # The logits stay fp32; the op only argmaxes over them.
    next_token_logits = torch.randn(
        bs, vocab_size, generator=g, device=device, dtype=torch.float32
    )
    positions = torch.randint(
        0, 4096, (bs,), generator=g, device=device, dtype=torch.int64
    )

    draft_tokens = None
    if with_draft_tokens:
        draft_tokens = torch.randint(
            0,
            vocab_size,
            (bs, NUM_DRAFT_COLS),
            generator=g,
            device=device,
            dtype=torch.int64,
        )

    return next_token_logits, positions, draft_tokens, draft_token_column


def build_inputs(binding, dtype, device):
    shape = (
        binding["bs"],
        binding["vocab_size"],
        binding["with_draft_tokens"],
        binding["draft_token_column"],
    )
    logits, positions, draft_tokens, column = build_case(shape, dtype, device)
    return (logits, positions), {
        "draft_tokens": draft_tokens,
        "draft_token_column": column,
    }


def key_shape(binding):
    return [
        binding["bs"],
        binding["vocab_size"],
        int(binding["with_draft_tokens"]),
        binding["draft_token_column"],
    ]


def config(binding, dtype):
    bs, V = binding["bs"], binding["vocab_size"]
    num_draft_cols = NUM_DRAFT_COLS if binding["with_draft_tokens"] else 0
    draft_token_column = binding["draft_token_column"]
    return {
        "inputs": {
            "next_token_logits": {"shape": [bs, V], "dtype": "torch.float32"},
            "positions": {
                "shape": [bs],
                "dtype": "torch.int64",
                "note": "原地写回",
            },
            "draft_tokens": (
                {"shape": [bs, num_draft_cols], "dtype": "torch.int64"}
                if num_draft_cols
                else {"scalar": None}
            ),
            "draft_token_column": {"scalar": draft_token_column},
        },
        "outputs": {
            "positions": {"shape": [bs], "dtype": "torch.int64"},
            "draft_tokens": {
                "shape": [bs, num_draft_cols],
                "dtype": "torch.int64",
            },
        },
        "dims": {
            "bs": bs,
            "vocab_size": V,
            "num_draft_cols": num_draft_cols,
            "draft_token_column": draft_token_column,
        },
    }
