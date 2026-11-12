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

"""Native SGLang collection workload for assign_draft_cache_locs_contiguous."""

import torch

OP_NAME = "assign_draft_cache_locs_contiguous"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (3,)

SHAPES = [(bs, 8, 5) for bs in (1, 8, 64, 512)]
MORE_SHAPES = [(bs, 8, 5) for bs in (2048,)]
_MAX_BATCH = 256
_POOL_LEN = 8192


def native():
    """解析裸 kernel 并返回按 args 计算 grid 的 wrapper。"""
    try:
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_draft_cache_locs_contiguous as _kernel,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _call(
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        topk,
        speculative_num_steps,
    ):
        _kernel[(req_pool_indices.shape[0],)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
        )
        return out_cache_loc

    return _call


def grid():
    return [
        {
            "bs": bs,
            "topk": topk,
            "steps": steps,
            **{"max_batch": _MAX_BATCH, "pool_len": _POOL_LEN},
        }
        for (bs, topk, steps) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, topk, steps = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (_MAX_BATCH, _POOL_LEN),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    req_pool_indices = torch.randperm(
        _MAX_BATCH, device="cpu", generator=perm_g
    )[:bs].to(device=device, dtype=torch.int32)
    seq_lens = torch.randint(
        0,
        _POOL_LEN - topk * steps,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    out_cache_loc = torch.zeros(
        bs * topk * steps, dtype=torch.int32, device=device
    )
    return req_pool_indices, req_to_token, seq_lens, out_cache_loc, topk, steps


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["topk"], binding["steps"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"bs{binding['bs']}_topk{binding['topk']}_steps{binding['steps']}"
        f"_mb{binding['max_batch']}_pl{binding['pool_len']}"
    )


def config(binding, dtype):
    bs, topk, steps = binding["bs"], binding["topk"], binding["steps"]
    max_batch, pool_len = binding["max_batch"], binding["pool_len"]
    return {
        "inputs": {
            "req_pool_indices": {
                "shape": [min(bs, max_batch)],
                "dtype": "torch.int32",
            },
            "req_to_token": {
                "shape": [max_batch, pool_len],
                "dtype": "torch.int32",
            },
            "seq_lens": {"shape": [bs], "dtype": "torch.int32"},
            "out_cache_loc": {
                "shape": [bs * topk * steps],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "topk": {"scalar": topk},
            "speculative_num_steps": {"scalar": steps},
        },
        "outputs": {
            "out_cache_loc": {
                "shape": [bs * topk * steps],
                "dtype": "torch.int32",
            },
        },
        "dims": {
            "bs": bs,
            "topk": topk,
            "steps": steps,
            "max_batch": max_batch,
            "pool_len": pool_len,
        },
    }
