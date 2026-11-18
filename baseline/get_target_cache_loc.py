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

"""Native SGLang collection workload for get_target_cache_loc."""

import torch

OP_NAME = "get_target_cache_loc"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(bs, 8) for bs in (1, 8, 64, 256)]
MORE_SHAPES = [(bs, 8) for bs in (1024,)]


def native():
    """解析裸 kernel 并返回复刻 baseline 启动（含 tgt/free 分配）的 wrapper。"""
    try:
        import triton
        from sglang.kernels.ops.speculative.cache_locs import (
            get_target_cache_loc as _kernel,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _call(
        num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
    ):
        bs = num_correct_drafts.shape[0]
        n_tgt = int(num_correct_drafts.sum()) + bs
        n_free = int(to_free_num_slots.sum())
        tgt = torch.zeros(
            max(n_tgt, 1),
            dtype=out_cache_loc.dtype,
            device=out_cache_loc.device,
        )
        free = torch.zeros(
            max(n_free, 1),
            dtype=out_cache_loc.dtype,
            device=out_cache_loc.device,
        )
        _kernel[(bs,)](
            tgt,
            free,
            num_correct_drafts,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
            triton.next_power_of_2(num_verify_tokens),
            triton.next_power_of_2(bs),
        )
        return tgt, free

    return _call


def grid():
    return [
        {"bs": bs, "num_verify_tokens": num_verify_tokens}
        for (bs, num_verify_tokens) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, num_verify_tokens = shape
    g = torch.Generator(device=device).manual_seed(9)
    num_correct_drafts = torch.randint(
        0,
        num_verify_tokens,
        (bs,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    to_free_num_slots = torch.randint(
        0,
        num_verify_tokens // 2 + 1,
        (bs,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    out_cache_loc = torch.randint(
        1,
        1 << 20,
        (bs * num_verify_tokens,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    return (
        num_correct_drafts,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["num_verify_tokens"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return f"bs{binding['bs']}_nvt{binding['num_verify_tokens']}"


def config(binding, dtype):
    bs, nvt = binding["bs"], binding["num_verify_tokens"]
    return {
        "inputs": {
            "num_correct_drafts": {"shape": [bs], "dtype": "torch.int64"},
            "to_free_num_slots": {"shape": [bs], "dtype": "torch.int64"},
            "out_cache_loc": {"shape": [bs * nvt], "dtype": "torch.int64"},
            "num_verify_tokens": {"scalar": nvt},
        },
        "outputs": {
            "tgt": {
                "note": "zeros(max(sum(num_correct_drafts)+bs, 1))，数据相关"
            },
            "free": {
                "note": "zeros(max(sum(to_free_num_slots), 1))，数据相关"
            },
        },
        "dims": {
            "bs": bs,
            "num_verify_tokens": nvt,
            "note": "tgt/free 长度由输入求和决定，数据相关",
        },
    }
