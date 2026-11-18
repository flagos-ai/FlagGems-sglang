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

"""Native SGLang collection workload for chunked_sgmv_shrink."""

import torch

if __package__:
    from .__lora_batch_utils import make_batch_info
else:
    from __lora_batch_utils import make_batch_info

OP_NAME = "chunked_sgmv_shrink"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (64, 8, 4, 32, 4096, 1),
    (256, 4, 2, 64, 4096, 1),
]
MORE_SHAPES = [
    (8, 1, 1, 16, 64, 1),
    (16, 4, 2, 16, 128, 1),
    (16, 2, 2, 32, 256, 3),
]


def native():
    """解析 chunked_sgmv_shrink.chunked_sgmv_lora_shrink_forward。"""
    try:
        from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
            chunked_sgmv_lora_shrink_forward as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def binding_for_case(shape):
    fields = ("seg_len", "num_segs", "num_lora", "r", "k", "num_slices")
    result = dict(zip(fields, shape))
    result["shape"] = shape
    result["seg_lens"] = [result["seg_len"]] * result["num_segs"]
    return result


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    seg_len, num_segs, num_lora, r, k, num_slices = shape
    g = torch.Generator(device=device).manual_seed(0)
    seg_lens = [seg_len] * num_segs
    s = sum(seg_lens)
    x = torch.randn(s, k, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    weights = torch.randn(
        num_lora,
        num_slices * r,
        k,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    weight_indices = [i % num_lora for i in range(num_segs)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        device=device,
        lora_ranks=[r] * num_lora,
        permutation="identity",
    )
    # batch_info is a dataclass, which unpack_to_args_kwargs drops from the
    # positional args -- it and every later parameter ride in a dict.
    return (
        x,
        weights,
        {
            "batch_info": batch_info,
            "num_slices": num_slices,
        },
    )


def build_inputs(binding, dtype, device):
    first, second, kwargs = build_case(binding["shape"], dtype, device)
    return (first, second, kwargs["batch_info"], kwargs["num_slices"]), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return (
        f"s{s}_L{binding['num_lora']}_r{binding['r']}_k{binding['k']}"
        f"_sl{binding['num_slices']}"
    )


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, k, num_slices = (
        binding["num_lora"],
        binding["r"],
        binding["k"],
        binding["num_slices"],
    )
    s = sum(seg_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, k], "dtype": dt},
            "weights": {"shape": [num_lora, num_slices * r, k], "dtype": dt},
            "batch_info": {
                "note": "LoRABatchInfo（seg_indptr/weight_indices/"
                "lora_ranks/scalings 等 LoRA 分段路由 metadata）"
            },
            "num_slices": {"scalar": num_slices},
        },
        "outputs": {
            "output": {"shape": [s, num_slices * r], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens),
            "s": s,
            "num_lora": num_lora,
            "r": r,
            "k": k,
            "num_slices": num_slices,
        },
    }
