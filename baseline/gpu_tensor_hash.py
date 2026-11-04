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

"""gpu_tensor_hash baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.memory.gpu_tensor_hash.gpu_tensor_hash
    调用形态复刻 problems/utility/gpu_tensor_hash/baseline.py：
        gpu_tensor_hash(tensor, seed=seed)
    产出标量/新张量，不写回入参 -> IS_INPLACE=False。

输入构造复刻 problems/utility/gpu_tensor_hash/cases.py 的 BENCH_CASES：
    tensor = randn(numel).bf16; seed = 0x243F6A88
    其中 numel in (1<<12, 1<<16, 1<<20, 1<<24)。
dtype 取 cases.py BENCH_CASES 的 [bfloat16]。
"""

import torch

OP_NAME = "gpu_tensor_hash"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SEED = 0x243F6A88
_NUMEL = [1 << 12, 1 << 16, 1 << 20, 1 << 24]


def native():
    """解析 sglang.kernels.ops.memory.gpu_tensor_hash.gpu_tensor_hash。"""
    try:
        from sglang.kernels.ops.memory.gpu_tensor_hash import gpu_tensor_hash as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"numel": n} for n in _NUMEL]


def build_inputs(binding, dtype, device):
    numel = binding["numel"]
    g = torch.Generator(device=device).manual_seed(9)
    t = torch.randn(numel, device=device, generator=g).to(dtype)
    return (t,), {"seed": _SEED}


def key_shape(binding):
    return [binding["numel"]]


def config(binding, dtype):
    numel = binding["numel"]
    dt = str(dtype)
    return {
        "inputs": {
            "tensor": {"shape": [numel], "dtype": dt},
            "seed": {"scalar": _SEED},
        },
        "outputs": {
            "hash": {"note": "标量哈希值（int）"},
        },
        "dims": {"numel": numel},
    }
