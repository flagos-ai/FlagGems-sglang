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

"""Native SGLang collection workload for chunked_sgmv_expand."""

import torch

if __package__:
    from .__lora_batch_utils import make_batch_info
else:
    from __lora_batch_utils import make_batch_info

OP_NAME = "chunked_sgmv_expand"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (5,)

SHAPES = [
    (64, 8, 4, 32, 4096, 4096),
    (256, 4, 2, 64, 4096, 1024, 1024),
]
MORE_SHAPES = [
    (8, 1, 1, 16, 64),
    (16, 4, 2, 16, 128, 64),
    (16, 2, 2, 32, 256, 128, 128),
]


def native():
    """解析 chunked_sgmv_expand.chunked_sgmv_lora_expand_forward。"""
    try:
        from sglang.kernels.ops.gemm.chunked_sgmv_expand import (
            chunked_sgmv_lora_expand_forward as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def binding_for_case(shape):
    fields = ("seg_len", "num_segs", "num_lora", "r")
    result = dict(zip(fields, shape))
    result["shape"] = shape
    result["seg_lens"] = [result["seg_len"]] * result["num_segs"]
    result["slice_sizes"] = shape[4:]
    return result


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    seg_len, num_segs, num_lora, r = shape[:4]
    slice_sizes = list(shape[4:])
    g = torch.Generator(device=device).manual_seed(0)
    seg_lens = [seg_len] * num_segs
    s = sum(seg_lens)
    n_slices = len(slice_sizes)
    output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)

    offsets = [0]
    for sz in slice_sizes:
        offsets.append(offsets[-1] + sz)
    slice_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    x = torch.randn(
        s,
        n_slices * r,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    weights = torch.randn(
        num_lora,
        output_dim,
        r,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    base_output = torch.randn(
        s, output_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    weight_indices = [i % num_lora for i in range(num_segs)]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        device=device,
        lora_ranks=[r] * num_lora,
        scalings=scalings,
        permutation="identity",
    )
    # batch_info is a dataclass, which unpack_to_args_kwargs drops from the
    # positional args -- it and every later parameter ride in a dict.
    return (
        x,
        weights,
        {
            "batch_info": batch_info,
            "slice_offsets": slice_offsets,
            "max_slice_size": max_slice_size,
            "base_output": base_output,
        },
    )


def build_inputs(binding, dtype, device):
    first, second, kwargs = build_case(binding["shape"], dtype, device)
    return (
        first,
        second,
        kwargs["batch_info"],
        kwargs["slice_offsets"],
        kwargs["max_slice_size"],
        kwargs["base_output"],
    ), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    ss = "x".join(str(z) for z in binding["slice_sizes"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_slc{ss}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, slice_sizes = (
        binding["num_lora"],
        binding["r"],
        binding["slice_sizes"],
    )
    s = sum(seg_lens)
    n_slices = len(slice_sizes)
    output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, n_slices * r], "dtype": dt},
            "weights": {"shape": [num_lora, output_dim, r], "dtype": dt},
            "batch_info": {
                "note": "LoRABatchInfo（seg_indptr/weight_indices/"
                "lora_ranks/scalings 等 LoRA 分段路由 metadata）"
            },
            "slice_offsets": {
                "shape": [n_slices + 1],
                "dtype": "torch.int32",
                "note": "slice_sizes 的前缀和偏移",
            },
            "max_slice_size": {"scalar": max_slice_size},
            "base_output": {
                "shape": [s, output_dim],
                "dtype": dt,
                "note": "原地写回",
            },
        },
        "outputs": {
            "base_output": {"shape": [s, output_dim], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens),
            "s": s,
            "num_lora": num_lora,
            "r": r,
            "slice_sizes": list(slice_sizes),
            "n_slices": n_slices,
            "output_dim": output_dim,
            "max_slice_size": max_slice_size,
        },
    }
