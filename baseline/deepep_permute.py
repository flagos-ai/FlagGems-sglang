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

"""deepep_permute baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.deepep_permute_triton_kernel 是
triton JIT kernel，以 grid=(input.shape[0],) 启动。复刻 problems/moe/
deepep_permute/baseline.py：out=gateup_input.clone()，启动 kernel(input, out,
src2dst, topk_ids, None, topk, hidden_size, BLOCK_SIZE=512)，原地写 out。
native() 返回做该次启动的包装 callable。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,2048}, topk=8, hidden=7168）：
    input=randn(t,hidden).bf16; gateup_input=zeros(t*topk,hidden).bf16;
    src2dst=randperm(t*topk).reshape(t,topk).int32; topk_ids=zeros(t,topk).int32。
IS_INPLACE=True（kernel 写回 out 缓冲）。
"""

import torch

OP_NAME = "deepep_permute"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_BLOCK_SIZE = 512
_SHAPES = [(t, 8, 7168) for t in (1, 8, 64, 512, 2048)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            deepep_permute_triton_kernel as _k,
        )
    except Exception:
        return None

    def _op(input, out, src2dst, topk_ids, topk, hidden_size):
        _k[(input.shape[0],)](
            input, out, src2dst, topk_ids, None, topk, hidden_size,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return out

    return _op


def grid():
    return [{"num_tokens": t, "topk": tk, "hidden": h} for (t, tk, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, topk, hidden = binding["num_tokens"], binding["topk"], binding["hidden"]
    n = num_tokens * topk
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(num_tokens, hidden, device=device, generator=g).to(dtype)
    out = torch.zeros(n, hidden, dtype=dtype, device=device)
    dst = torch.randperm(n, device=device, generator=g).reshape(num_tokens, topk).to(torch.int32)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    return (x, out, dst, topk_ids, topk, hidden), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_topk{binding['topk']}_h{binding['hidden']}"


def config(binding, dtype):
    num_tokens, topk, hidden = binding["num_tokens"], binding["topk"], binding["hidden"]
    n = num_tokens * topk
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [num_tokens, hidden], "dtype": dt},
            "out": {"shape": [n, hidden], "dtype": dt, "note": "原地写回（gateup_input.clone()）"},
            "src2dst": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk_ids": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk": {"scalar": topk},
            "hidden_size": {"scalar": hidden},
            "BLOCK_SIZE": {"scalar": _BLOCK_SIZE},
        },
        "outputs": {
            "out": {"shape": [n, hidden], "dtype": dt},
        },
        "dims": {"num_tokens": num_tokens, "topk": topk, "hidden": hidden, "n": n},
    }
