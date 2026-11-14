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

# Uniform gather semantics: SGLang (Apache-2.0), commit c07883d.
import torch
import triton
import triton.language as tl

__all__ = ["assign_extend_cache_locs_uniform"]


@triton.jit
def _flat(
    I,  # noqa: E741
    P,
    S,
    O,  # noqa: E741
    N: tl.constexpr,
    D: tl.constexpr,
    STRIDES: tl.constexpr,
):
    IS: tl.constexpr = STRIDES[0]
    RS: tl.constexpr = STRIDES[1]
    CS: tl.constexpr = STRIDES[2]
    SS: tl.constexpr = STRIDES[3]
    B: tl.constexpr = 256
    q = tl.program_id(0) * B + tl.arange(0, B)
    r = q // D
    c = q % D
    live = q < N * D
    pool = tl.load(I + r * IS, live, 0).to(tl.int64)
    start = tl.load(S + r * SS, live, 0).to(tl.int64)
    value = tl.load(P + pool * RS + (start + c) * CS, live)
    tl.store(O + q, value.to(tl.int64), live)


def assign_extend_cache_locs_uniform(
    req_pool_indices,
    req_to_token,
    start_offset,
    batch_size,
    draft_token_num,
    device,
):
    total = batch_size * draft_token_num
    out = torch.empty((total,), dtype=torch.int64, device=device)
    if total:
        _flat[(triton.cdiv(total, 256),)](
            req_pool_indices,
            req_to_token,
            start_offset,
            out,
            batch_size,
            draft_token_num,
            (
                req_pool_indices.stride(0),
                *req_to_token.stride(),
                start_offset.stride(0),
            ),
            num_warps=4,
        )
    return out
