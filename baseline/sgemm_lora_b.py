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

"""Native SGLang collection workload for sgemm_lora_b."""

import torch

if __package__:
    from .__lora_batch_utils import make_batch_info
else:
    from __lora_batch_utils import make_batch_info

OP_NAME = "sgemm_lora_b"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (3,)

SHAPES = [
    ((64,) * 8, 4, 32, 4096),
    ((256,) * 4, 2, 64, 4096),
]
MORE_SHAPES = [
    ((5,), 1, 16, 64),
    ((3, 7, 0, 12), 2, 16, 128),
    ((9, 4), 2, 32, 256),
]


def native():
    """解析 sglang.kernels.ops.gemm.sgemm_lora_b.sgemm_lora_b_fwd。"""
    try:
        from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def binding_for_case(shape):
    fields = ("seg_lens", "num_lora", "r", "n")
    result = dict(zip(fields, shape))
    result["shape"] = shape
    return result


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    seg_lens, num_lora, r, n = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)
    x = torch.randn(s, r, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    weights = torch.randn(
        num_lora, n, r, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    base_output = torch.randn(
        s, n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        device=device,
        lora_ranks=[r] * num_lora,
        scalings=[0.5 + 0.25 * i for i in range(num_lora)],
    )
    # ``batch_info`` is a dataclass, which unpack_to_args_kwargs would drop
    # from the positional args, so route it (and the tail) through kwargs.
    return x, weights, dict(batch_info=batch_info, base_output=base_output)


def build_inputs(binding, dtype, device):
    first, second, kwargs = build_case(binding["shape"], dtype, device)
    return (first, second, kwargs["batch_info"], kwargs["base_output"]), {}


def key_shape(binding):
    s = sum(binding["seg_lens"])
    return f"s{s}_L{binding['num_lora']}_r{binding['r']}_n{binding['n']}"


def config(binding, dtype):
    seg_lens = binding["seg_lens"]
    num_lora, r, n = binding["num_lora"], binding["r"], binding["n"]
    s = sum(seg_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [s, r], "dtype": dt},
            "weights": {"shape": [num_lora, n, r], "dtype": dt},
            "batch_info": {
                "note": "LoRABatchInfo（seg_indptr/weight_indices/"
                "lora_ranks/scalings 等 LoRA 分段路由 metadata）"
            },
            "base_output": {"shape": [s, n], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "base_output": {"shape": [s, n], "dtype": dt},
        },
        "dims": {
            "seg_lens": list(seg_lens),
            "s": s,
            "num_lora": num_lora,
            "r": r,
            "n": n,
        },
    }
