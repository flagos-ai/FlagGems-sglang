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

"""ep_gather baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.ep_gather（destination-passing）。
复刻 problems/moe/ep_gather/baseline.py：output_tensor=output_tensor.clone()，
调 ep_gather(input_tensor, recv_topk_ids, recv_topk_weight, input_index,
output_tensor)，原地写 output_tensor 并返回它。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, hidden=4096, topk=8,
num_src=max(t*8,16)）：input_tensor=randn(num_src,hidden).bf16;
recv_topk_ids=randint(0,32).int32; recv_topk_weight=rand.fp32;
input_index=randint(0,num_src).int32; output_tensor=full(t,hidden,5.0).bf16。
IS_INPLACE=True（kernel 写回 output_tensor）。
"""

import torch

OP_NAME = "ep_gather"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(t, 4096, 8, max(t * 8, 16)) for t in (1, 8, 64, 512, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "hidden": h, "topk": tk, "num_src": ns}
        for (t, h, tk, ns) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    num_tokens, hidden, topk, num_src = (
        binding["num_tokens"], binding["hidden"], binding["topk"], binding["num_src"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    input_tensor = torch.randn(
        num_src, hidden, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    ids = torch.randint(
        0, 32, (num_tokens, topk), device=device, generator=g, dtype=torch.int32
    ).contiguous()
    weight = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    ).contiguous()
    index = torch.randint(
        0, num_src, (num_tokens, topk), device=device, generator=g, dtype=torch.int32
    ).contiguous()
    out = torch.full(
        (num_tokens, hidden), 5.0, dtype=torch.float32, device=device
    ).to(dtype).contiguous()
    return (input_tensor, ids, weight, index, out), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_h{binding['hidden']}_topk{binding['topk']}_src{binding['num_src']}"


def config(binding, dtype):
    num_tokens, hidden, topk, num_src = (
        binding["num_tokens"], binding["hidden"], binding["topk"], binding["num_src"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "input_tensor": {"shape": [num_src, hidden], "dtype": dt},
            "recv_topk_ids": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "recv_topk_weight": {"shape": [num_tokens, topk], "dtype": "torch.float32"},
            "input_index": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "output_tensor": {"shape": [num_tokens, hidden], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "output_tensor": {"shape": [num_tokens, hidden], "dtype": dt},
        },
        "dims": {"num_tokens": num_tokens, "hidden": hidden, "topk": topk, "num_src": num_src},
    }
