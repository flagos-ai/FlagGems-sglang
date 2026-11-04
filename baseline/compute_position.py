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

"""compute_position baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.position.compute_position_triton
    调用形态复刻 problems/attention/compute_position/baseline.py：
        compute_position_triton(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)
    返回新张量（positions, extend_start_loc），不原地写回；IS_INPLACE=False。

输入构造复刻 problems/attention/compute_position/cases.py 的 BENCH_CASES：
    seq_lens = randint(1, max_len+1, (bs,), int32)，max_len=512，with_prefix=True
    prefix = randint(0, 4096, (bs,), int32)，extend_seq_lens_sum = int(seq_lens.sum())。
    seed=7。
dtype：输入为 int32 索引张量，DTYPES 取 [int32]。
"""

import torch

OP_NAME = "compute_position"
DTYPES = [torch.int32]
IS_INPLACE = False

_MAX_LEN = 512
_CASES = [1, 8, 64, 256, 1024]


def native():
    """解析 sglang.kernels.ops.attention.position.compute_position_triton。"""
    try:
        from sglang.kernels.ops.attention.position import compute_position_triton as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "max_len": _MAX_LEN} for bs in _CASES]


def build_inputs(binding, dtype, device):
    bs, max_len = binding["bs"], binding["max_len"]
    g = torch.Generator(device=device).manual_seed(7)
    seq_lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    prefix = torch.randint(0, 4096, (bs,), dtype=torch.int32, device=device, generator=g)
    extend_seq_lens_sum = int(seq_lens.sum())
    return (prefix, seq_lens, extend_seq_lens_sum), {}


def key_shape(binding):
    return [binding["bs"]]


def config(binding, dtype):
    bs = binding["bs"]
    return {
        "inputs": {
            "extend_prefix_lens": {"shape": [bs], "dtype": "torch.int32"},
            "extend_seq_lens": {"shape": [bs], "dtype": "torch.int32"},
            "extend_seq_lens_sum": {"note": "标量 int = sum(extend_seq_lens)，数据相关"},
        },
        "outputs": {
            "positions": {"note": "positions 向量，长度=extend_seq_lens_sum，数据相关"},
            "extend_start_loc": {"shape": [bs], "dtype": "torch.int32"},
        },
        "dims": {"bs": bs, "max_len": binding["max_len"]},
    }
