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

"""chain_speculative_sampling baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton
    调用形态复刻 problems/sampling_grammar/chain_speculative_sampling/baseline.py：
        predicts/accept_index/accept_token_num 为 native 原地写入的输出缓冲，
        另造 dummy_next_token/dummy_next_sibling，末三参 (0.0, 0.0, False)。
    native 把结果写进预分配的 predicts/accept_index/accept_token_num 缓冲；
    这些缓冲由 build_inputs 每次重建，故 IS_INPLACE=True。

输入构造复刻 problems/sampling_grammar/chain_speculative_sampling/cases.py 的 BENCH_CASES：
    _case(b, 4, 32000)：candidates=randint(0,V,(B,S)) int32；
    retrive_index=arange(B*S).reshape(B,S) int64；
    uniform_samples=rand(B,S-1)；uniform_samples_for_final_sampling=rand(B)；
    target_probs=softmax(randn(B,S,V))；draft_probs=softmax(randn(B,S-1,V))；num_slots=B*S。
dtype 取 benchmark/test_chain_speculative_sampling.py 的 [float32]。
"""

import torch

OP_NAME = "chain_speculative_sampling"
DTYPES = [torch.float32]
IS_INPLACE = True  # native 写入预分配的 predicts/accept_index/accept_token_num 缓冲

# BENCH_CASES: _case(b, 4, 32000) for b in (1,8,64,512,4096)
_SHAPES = [{"B": b, "S": 4, "V": 32000} for b in (1, 8, 64, 512, 4096)]


def native():
    """解析 chain_speculative_sampling_triton。"""
    try:
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    B, S, V = binding["B"], binding["S"], binding["V"]
    g = torch.Generator(device=device).manual_seed(0)

    candidates = torch.randint(0, V, (B, S), dtype=torch.int32, device=device, generator=g)
    retrive_index = torch.arange(B * S, dtype=torch.int64, device=device).reshape(B, S)
    uniform_samples = torch.rand(B, S - 1, device=device, generator=g)
    uniform_samples_for_final_sampling = torch.rand(B, device=device, generator=g)
    target_probs = torch.softmax(
        torch.randn(B, S, V, device=device, generator=g), dim=-1
    )
    draft_probs = torch.softmax(
        torch.randn(B, S - 1, V, device=device, generator=g), dim=-1
    )
    num_slots = B * S

    # native 原地写入的输出缓冲 + dummy 树结构（复刻 baseline.py）。
    predicts = torch.zeros(num_slots, dtype=candidates.dtype, device=device)
    accept_index = torch.full(
        (B, S), -1, dtype=retrive_index.dtype, device=device
    )
    accept_token_num = torch.zeros(B, dtype=torch.int32, device=device)
    dummy_next_token = torch.zeros_like(candidates)
    dummy_next_sibling = torch.zeros_like(candidates)

    args = (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        dummy_next_token,
        dummy_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        0.0,
        0.0,
        False,
    )
    return args, {}


def key_shape(binding):
    return "B{B}_S{S}_V{V}".format(**binding)


def config(binding, dtype):
    B, S, V = binding["B"], binding["S"], binding["V"]
    num_slots = B * S
    dt = str(dtype)
    return {
        "inputs": {
            "predicts": {"shape": [num_slots], "dtype": "torch.int32",
                         "note": "原地写回（输出缓冲）"},
            "accept_index": {"shape": [B, S], "dtype": "torch.int64",
                             "note": "原地写回（输出缓冲）"},
            "accept_token_num": {"shape": [B], "dtype": "torch.int32",
                                 "note": "原地写回（输出缓冲）"},
            "candidates": {"shape": [B, S], "dtype": "torch.int32"},
            "retrive_index": {"shape": [B, S], "dtype": "torch.int64"},
            "retrive_next_token": {"shape": [B, S], "dtype": "torch.int32",
                                   "note": "dummy 树结构"},
            "retrive_next_sibling": {"shape": [B, S], "dtype": "torch.int32",
                                     "note": "dummy 树结构"},
            "uniform_samples": {"shape": [B, S - 1], "dtype": dt},
            "uniform_samples_for_final_sampling": {"shape": [B], "dtype": dt},
            "target_probs": {"shape": [B, S, V], "dtype": dt},
            "draft_probs": {"shape": [B, S - 1, V], "dtype": dt},
            "threshold_single": {"scalar": 0.0},
            "threshold_acc": {"scalar": 0.0},
            "deterministic": {"scalar": False},
        },
        "outputs": {
            "predicts": {"shape": [num_slots], "dtype": "torch.int32"},
            "accept_index": {"shape": [B, S], "dtype": "torch.int64"},
            "accept_token_num": {"shape": [B], "dtype": "torch.int32"},
        },
        "dims": {"B": B, "S": S, "V": V, "num_slots": num_slots},
    }
