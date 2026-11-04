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

"""compute_src2dst baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.compute_src2dst_triton_kernel
是一个 triton JIT kernel，需用 grid 启动。复刻 problems/moe/compute_src2dst/
baseline.py：分配 src2dst=empty(num_toks,int32)，以 grid=cdiv(num_toks,512)
启动 kernel(reorder_ids, src2dst, num_toks, BLOCK_SIZE=512)，原地写 src2dst。
native() 返回一个做该次 grid 启动的包装 callable，供采集器 op(*args) 调用。

输入复刻 cases.py BENCH_CASES：reorder_ids=randperm(num_toks)（int64）。
IS_INPLACE=True（kernel 写回 src2dst 缓冲）。
"""

import torch

OP_NAME = "compute_src2dst"
DTYPES = [torch.int64]
IS_INPLACE = True

_BLOCK_SIZE = 512
_SHAPES = [8, 512, 8192, 131072, 1048576]


def native():
    try:
        import triton
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            compute_src2dst_triton_kernel as _k,
        )
    except Exception:
        return None

    def _op(reorder_ids, src2dst, num_toks, block_size):
        _k[(triton.cdiv(num_toks, block_size),)](
            reorder_ids, src2dst, num_toks, block_size
        )
        return src2dst

    return _op


def grid():
    return [{"num_toks": n} for n in _SHAPES]


def build_inputs(binding, dtype, device):
    num_toks = binding["num_toks"]
    g = torch.Generator(device=device).manual_seed(0)
    reorder_ids = torch.randperm(num_toks, device=device, generator=g)
    src2dst = torch.empty(num_toks, dtype=torch.int32, device=device)
    return (reorder_ids, src2dst, num_toks, _BLOCK_SIZE), {}


def key_shape(binding):
    return [binding["num_toks"]]


def config(binding, dtype):
    n = binding["num_toks"]
    return {
        "inputs": {
            "reorder_ids": {"shape": [n], "dtype": "torch.int64"},
            "src2dst": {"shape": [n], "dtype": "torch.int32", "note": "原地写回"},
            "num_toks": {"scalar": n},
            "BLOCK_SIZE": {"scalar": _BLOCK_SIZE},
        },
        "outputs": {
            "src2dst": {"shape": [n], "dtype": "torch.int32"},
        },
        "dims": {"num_toks": n},
    }
