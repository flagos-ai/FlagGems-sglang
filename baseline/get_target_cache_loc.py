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

"""get_target_cache_loc baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.cache_locs.get_target_cache_loc
    是裸 triton kernel，baseline.py 按 bs 启动并分配 data-dependent 输出：
        n_tgt = int(num_correct_drafts.sum()) + bs; n_free = int(to_free_num_slots.sum())
        tgt = zeros(max(n_tgt,1)); free = zeros(max(n_free,1))
        get_target_cache_loc[(bs,)](tgt, free, num_correct_drafts,
            to_free_num_slots, out_cache_loc, num_verify_tokens,
            next_power_of_2(num_verify_tokens), next_power_of_2(bs))
        return tgt, free
    采集器只 op(*args)，native() 返回复刻该启动并分配 tgt/free 的 wrapper。
    输出为新分配张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    num_correct_drafts = randint(0, nvt, (bs,)) int64
    to_free_num_slots  = randint(0, nvt//2+1, (bs,)) int64
    out_cache_loc = randint(1, 1<<20, (bs*nvt,)) int64
    其中 num_verify_tokens(nvt)=8，bs in (1,8,64,256,1024)。
全整型，DTYPES 记 int64。
"""

import torch

OP_NAME = "get_target_cache_loc"
DTYPES = [torch.int64]
IS_INPLACE = False

_NVT = 8
_BS = [1, 8, 64, 256, 1024]


def native():
    """解析裸 kernel 并返回复刻 baseline 启动（含 tgt/free 分配）的 wrapper。"""
    try:
        import triton

        from sglang.kernels.ops.speculative.cache_locs import (
            get_target_cache_loc as _kernel,
        )
    except Exception:
        return None

    def _call(num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens):
        bs = num_correct_drafts.shape[0]
        n_tgt = int(num_correct_drafts.sum()) + bs
        n_free = int(to_free_num_slots.sum())
        tgt = torch.zeros(
            max(n_tgt, 1), dtype=out_cache_loc.dtype, device=out_cache_loc.device
        )
        free = torch.zeros(
            max(n_free, 1), dtype=out_cache_loc.dtype, device=out_cache_loc.device
        )
        _kernel[(bs,)](
            tgt,
            free,
            num_correct_drafts,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
            triton.next_power_of_2(num_verify_tokens),
            triton.next_power_of_2(bs),
        )
        return tgt, free

    return _call


def grid():
    return [{"bs": b, "num_verify_tokens": _NVT} for b in _BS]


def build_inputs(binding, dtype, device):
    bs, nvt = binding["bs"], binding["num_verify_tokens"]
    g = torch.Generator(device=device).manual_seed(9)
    correct = torch.randint(
        0, nvt, (bs,), dtype=torch.int64, device=device, generator=g
    )
    free = torch.randint(
        0, nvt // 2 + 1, (bs,), dtype=torch.int64, device=device, generator=g
    )
    out_cache_loc = torch.randint(
        1, 1 << 20, (bs * nvt,), dtype=torch.int64, device=device, generator=g
    )
    return (correct, free, out_cache_loc, nvt), {}


def key_shape(binding):
    return (f"bs{binding['bs']}_nvt{binding['num_verify_tokens']}")


def config(binding, dtype):
    bs, nvt = binding["bs"], binding["num_verify_tokens"]
    return {
        "inputs": {
            "num_correct_drafts": {"shape": [bs], "dtype": "torch.int64"},
            "to_free_num_slots": {"shape": [bs], "dtype": "torch.int64"},
            "out_cache_loc": {"shape": [bs * nvt], "dtype": "torch.int64"},
            "num_verify_tokens": {"scalar": nvt},
        },
        "outputs": {
            "tgt": {"note": "zeros(max(sum(num_correct_drafts)+bs, 1))，数据相关"},
            "free": {"note": "zeros(max(sum(to_free_num_slots), 1))，数据相关"},
        },
        "dims": {
            "bs": bs, "num_verify_tokens": nvt,
            "note": "tgt/free 长度由输入求和决定，数据相关",
        },
    }
