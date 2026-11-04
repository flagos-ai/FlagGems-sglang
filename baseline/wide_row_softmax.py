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

"""wide_row_softmax baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.multi_layer_eagle.wide_row_softmax_triton
    调用形态复刻 problems/speculative/wide_row_softmax/baseline.py：
        out = out.clone(); return wide_row_softmax_triton(logits, temperatures, out)
    kernel 把 softmax 结果写入 out 并返回 -> 原地写回 out 实参。
    IS_INPLACE=True（采集侧每次重建 out）。

输入构造复刻 cases.py 的 BENCH_CASES：
    logits = (randn(bs, vocab) fp32) * 4
    out = full((bs, out_stride), -1.0) fp32（out_stride=vocab）
    temperatures = None
    其中 bs in (1,4,32)，vocab in (32000, 131072)。
dtype：fp32。
"""

import torch

OP_NAME = "wide_row_softmax"
DTYPES = [torch.float32]
IS_INPLACE = True  # kernel 把结果写入 out 实参

_BS = [1, 4, 32]
_VOCAB = [32000, 131072]


def native():
    """解析 wide_row_softmax_triton。"""
    try:
        from sglang.kernels.ops.speculative.multi_layer_eagle import (
            wide_row_softmax_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": b, "vocab": v} for b in _BS for v in _VOCAB]


def build_inputs(binding, dtype, device):
    bs, vocab = binding["bs"], binding["vocab"]
    out_stride = vocab
    g = torch.Generator(device=device).manual_seed(0)
    logits = (
        torch.randn(bs, vocab, dtype=torch.float32, device=device, generator=g) * 4
    ).contiguous()
    out = torch.full((bs, out_stride), -1.0, dtype=torch.float32, device=device)
    return (logits, None, out), {}


def key_shape(binding):
    return [binding["bs"], binding["vocab"]]


def config(binding, dtype):
    bs, vocab = binding["bs"], binding["vocab"]
    return {
        "inputs": {
            "logits": {"shape": [bs, vocab], "dtype": "torch.float32"},
            "temperatures": {"scalar": None},
            "out": {"shape": [bs, vocab], "dtype": "torch.float32",
                    "note": "原地写回（out_stride=vocab）"},
        },
        "outputs": {
            "out": {"shape": [bs, vocab], "dtype": "torch.float32"},
        },
        "dims": {"bs": bs, "vocab": vocab, "out_stride": vocab},
    }
