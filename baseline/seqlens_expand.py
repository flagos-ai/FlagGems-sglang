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

"""Native SGLang collection workload for seqlens_expand."""

import torch

OP_NAME = "seqlens_expand"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

MAX_Q = 8

SHAPES = [(n,) for n in (1, 8, 64, 512)]
MORE_SHAPES = [(n,) for n in (4096,)]


def native():
    """解析 sglang.kernels.ops.attention.pad.seqlens_expand_triton。"""
    try:
        from sglang.kernels.ops.attention import pad
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    op = getattr(pad, "seqlens_expand_triton", None)
    return op if callable(op) else None


def grid():
    return [{"n": n, **{"max_q": MAX_Q}} for (n,) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    (n,) = shape
    max_q = MAX_Q
    g = torch.Generator(device=device).manual_seed(0)
    extend_seq_lens = torch.randint(
        1, max_q + 1, (n,), dtype=torch.int32, device=device, generator=g
    )
    seq_lens = torch.randint(
        1, 8192, (n,), dtype=torch.int32, device=device, generator=g
    )
    return (
        extend_seq_lens,
        seq_lens,
        int(extend_seq_lens.sum()),
        int(extend_seq_lens.max()),
    )


def build_inputs(binding, dtype, device):
    shape = (binding["n"],)
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["n"]]


def config(binding, dtype):
    n, max_q = binding["n"], binding["max_q"]
    return {
        "inputs": {
            "extend_seq_lens": {"shape": [n], "dtype": "torch.int32"},
            "seq_lens": {"shape": [n], "dtype": "torch.int32"},
            "total_len": {"note": "标量 int = sum(extend_seq_lens)，数据相关"},
            "max_q_len": {"note": "标量 int = max(extend_seq_lens)，数据相关"},
        },
        "outputs": {
            "out": {
                "shape": ["total_len"],
                "dtype": "torch.int32",
                "note": "按 qo 展开，长度 total_len 数据相关",
            },
        },
        "dims": {
            "n": n,
            "max_q": max_q,
            "note": "total_len/max_q_len 数据相关（seed=0）",
        },
    }
