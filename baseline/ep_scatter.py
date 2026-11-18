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

"""Native SGLang collection workload for ep_scatter."""

import torch

OP_NAME = "ep_scatter"
DTYPES = [torch.float8_e4m3fn]
IS_INPLACE = True

MUTATED_INPUTS = (5, 6, 7, 8, 9)

SHAPES = [(t, 8, 2048, 8) for t in (64, 512)]
MORE_SHAPES = [(4096, 8, 2048, 8)]
_BLOCK_E = 128
_QUANT_BLOCK = 128


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"t": t, "topk": topk, "hidden": hidden, "num_experts": num_experts}
        for (t, topk, hidden, num_experts) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    t, topk, hidden, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    ids = torch.randint(
        0,
        num_experts,
        (t, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    valid = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        valid[e] = int((ids == e).sum())
    padded = ((valid + _BLOCK_E - 1) // _BLOCK_E * _BLOCK_E).to(torch.int32)
    total = int(padded.sum())
    recv_x = torch.randn(
        t, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    recv_x_scale = torch.rand(
        t,
        hidden // _QUANT_BLOCK,
        dtype=torch.float32,
        device=device,
        generator=g,
    )
    return (
        recv_x,
        recv_x_scale,
        ids,
        padded,
        valid,
        torch.zeros(num_experts, dtype=torch.int32, device=device),
        torch.zeros((total, hidden), dtype=cur_dtype, device=device),
        torch.zeros(
            (total, hidden // _QUANT_BLOCK),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(total, dtype=torch.int32, device=device),
        torch.full((t, topk), -1, dtype=torch.int32, device=device),
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["t"],
        binding["topk"],
        binding["hidden"],
        binding["num_experts"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return f"t{binding['t']}_topk{binding['topk']}_h{binding['hidden']}_E{binding['num_experts']}"


def config(binding, dtype):
    t, topk, hidden, num_experts = (
        binding["t"],
        binding["topk"],
        binding["hidden"],
        binding["num_experts"],
    )
    dt = str(dtype)
    g = hidden // _QUANT_BLOCK
    return {
        "inputs": {
            "recv_x": {"shape": [t, hidden], "dtype": dt},
            "recv_x_scale": {"shape": [t, g], "dtype": "torch.float32"},
            "recv_topk": {"shape": [t, topk], "dtype": "torch.int32"},
            "num_recv_tokens_per_expert": {
                "shape": [num_experts],
                "dtype": "torch.int32",
            },
            "num_valid_tokens_per_expert": {
                "shape": [num_experts],
                "dtype": "torch.int32",
            },
            "expert_start_loc": {
                "shape": [num_experts],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "output_tensor": {
                "shape": ["total", hidden],
                "dtype": dt,
                "note": "原地写回",
            },
            "output_tensor_scale": {
                "shape": ["total", g],
                "dtype": "torch.float32",
                "note": "原地写回",
            },
            "m_indices": {
                "shape": ["total"],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "output_index": {
                "shape": [t, topk],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
        },
        "outputs": {
            "expert_start_loc": {
                "shape": [num_experts],
                "dtype": "torch.int32",
            },
            "output_tensor": {"shape": ["total", hidden], "dtype": dt},
            "output_tensor_scale": {
                "shape": ["total", g],
                "dtype": "torch.float32",
            },
            "m_indices": {"shape": ["total"], "dtype": "torch.int32"},
            "output_index": {"shape": [t, topk], "dtype": "torch.int32"},
        },
        "dims": {
            "t": t,
            "topk": topk,
            "hidden": hidden,
            "num_experts": num_experts,
            "BLOCK_E": _BLOCK_E,
            "quant_block": _QUANT_BLOCK,
            "note": "total = sum(pad(per-expert counts, BLOCK_E))，随机 ids 决定，数据相关",
        },
    }
