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

"""clamp_position baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.clamp_position.clamp_position_cuda
    调用形态复刻 problems/attention/clamp_position/baseline.py：
        clamp_position_cuda(seq_lens)
    返回新张量，不原地写回；IS_INPLACE=False。

输入构造复刻 problems/attention/clamp_position/cases.py 的 BENCH_CASES：
    seq_lens = randint(0, 8192, (bs,), int32)。seed=0。
dtype：该算子输入为 int32 索引张量（非遍历浮点 dtype），DTYPES 取 [int32]。
"""

import torch

OP_NAME = "clamp_position"
DTYPES = [torch.int32]
IS_INPLACE = False

_CASES = [1, 8, 64, 512, 4096, 16384]


def native():
    """解析 sglang.kernels.ops.attention.clamp_position.clamp_position_cuda。"""
    try:
        from sglang.kernels.ops.attention.clamp_position import clamp_position_cuda as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs} for bs in _CASES]


def build_inputs(binding, dtype, device):
    bs = binding["bs"]
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = torch.randint(0, 8192, (bs,), dtype=torch.int32, device=device, generator=g)
    return (seq_lens,), {}


def key_shape(binding):
    return [binding["bs"]]


def config(binding, dtype):
    bs = binding["bs"]
    return {
        "inputs": {
            "seq_lens": {"shape": [bs], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [bs], "dtype": "torch.int32"},
        },
        "dims": {"bs": bs},
    }
