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

"""Native SGLang collection workload for qkv_lora_b."""

import torch

if __package__:
    from .__lora_batch_utils import make_batch_info
else:
    from __lora_batch_utils import make_batch_info

OP_NAME = "qkv_lora_b"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (5,)

SHAPES = [
    ((64,) * 8, 4, 32, 4096, 1024),
    ((256,) * 4, 2, 64, 4096, 1024),
]
MORE_SHAPES = [
    ((5,), 1, 16, 64, 32),
    ((3, 7, 0, 12), 2, 16, 128, 64),
    ((9, 4), 2, 32, 256, 128),
]


def native():
    """解析 sglang.kernels.ops.gemm.qkv_lora_b.qkv_lora_b_fwd。"""
    try:
        from sglang.kernels.ops.gemm.qkv_lora_b import qkv_lora_b_fwd as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def binding_for_case(shape):
    fields = ("seg_lens", "num_lora", "r", "dq", "dkv")
    result = dict(zip(fields, shape))
    result["shape"] = shape
    return result


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    seg_lens, num_lora, r, dq, dkv = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)
    output_dim = dq + 2 * dkv
    x = torch.randn(
        s, 3 * r, generator=g, device=device, dtype=torch.float32
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
    output_offset = torch.tensor(
        [0, dq, dq + dkv, output_dim], dtype=torch.int32, device=device
    )
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        device=device,
        lora_ranks=[r] * num_lora,
        scalings=[0.5 + 0.25 * i for i in range(num_lora)],
    )
    # ``batch_info`` is a dataclass, which unpack_to_args_kwargs would drop
    # from the positional args, so route it (and the tail) through kwargs.
    return (
        x,
        weights,
        dict(
            batch_info=batch_info,
            output_offset=output_offset,
            max_qkv_out_dim=max(dq, dkv),
            base_output=base_output,
        ),
    )


def build_inputs(binding, dtype, device):
    first, second, kwargs = build_case(binding["shape"], dtype, device)
    return (
        first,
        second,
        kwargs["batch_info"],
        kwargs["output_offset"],
        kwargs["max_qkv_out_dim"],
        kwargs["base_output"],
    ), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_dq{binding['dq']}_dkv{binding['dkv']}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, dq, dkv = (
        binding["num_lora"],
        binding["r"],
        binding["dq"],
        binding["dkv"],
    )
    s = sum(seg_lens)
    output_dim = dq + 2 * dkv
    max_qkv_out_dim = max(dq, dkv)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, 3 * r], "dtype": dt},
            "qkv_lora_b": {"shape": [num_lora, output_dim, r], "dtype": dt},
            "batch_info": {
                "note": "LoRABatchInfo（seg_indptr/weight_indices/"
                "lora_ranks/scalings 等 LoRA 分段路由 metadata）"
            },
            "output_offset": {
                "shape": [4],
                "dtype": "torch.int32",
                "note": "[0, dq, dq+dkv, dq+2*dkv] 的 qkv 切片偏移",
            },
            "max_qkv_out_dim": {"scalar": max_qkv_out_dim},
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
            "dq": dq,
            "dkv": dkv,
            "output_dim": output_dim,
            "max_qkv_out_dim": max_qkv_out_dim,
        },
    }
