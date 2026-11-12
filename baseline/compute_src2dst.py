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

"""Native SGLang collection workload for compute_src2dst."""

import torch

OP_NAME = "compute_src2dst"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (1,)

_BLOCK_SIZE = 512

SHAPES = [(n,) for n in (8, 512, 8192, 131072)]
MORE_SHAPES = [(n,) for n in (1048576,)]


def native():
    try:
        import triton
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            compute_src2dst_triton_kernel as _k,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _op(reorder_ids, src2dst, num_toks, block_size):
        _k[(triton.cdiv(num_toks, block_size),)](
            reorder_ids, src2dst, num_toks, block_size
        )
        return src2dst

    return _op


def grid():
    return [{"num_toks": n} for (n,) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    (n,) = shape
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    g = torch.Generator(device="cpu").manual_seed(0)
    reorder_ids = torch.randperm(n, generator=g).to(
        dtype=torch.int32, device=device
    )
    return (reorder_ids, n)


def build_inputs(binding, dtype, device):
    shape = (binding["num_toks"],)
    ids, num_tokens = build_case(shape, dtype, device)
    out = torch.zeros_like(ids, dtype=torch.int32)
    return (ids, out, num_tokens, _BLOCK_SIZE), {}


def key_shape(binding):
    return [binding["num_toks"]]


def config(binding, dtype):
    n = binding["num_toks"]
    return {
        "inputs": {
            "reorder_ids": {"shape": [n], "dtype": "torch.int32"},
            "src2dst": {
                "shape": [n],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "num_toks": {"scalar": n},
            "BLOCK_SIZE": {"scalar": _BLOCK_SIZE},
        },
        "outputs": {
            "src2dst": {"shape": [n], "dtype": "torch.int32"},
        },
        "dims": {"num_toks": n},
    }
