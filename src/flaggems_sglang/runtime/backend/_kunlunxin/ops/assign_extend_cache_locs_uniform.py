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
def _fullrows_element(
    I,  # noqa: E741
    P,
    S,
    O,  # noqa: E741
    N: tl.constexpr,
    D: tl.constexpr,
    IS: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    C: tl.constexpr,
):
    q = tl.arange(0, 64 * C)
    global_q = tl.program_id(0) * (64 * C) + q
    r = global_q // D
    c = global_q % D
    live = global_q < N * D
    pool = tl.load(I + r * IS, live).to(tl.int64)
    start = tl.load(S + r * SS, live).to(tl.int64)
    value = tl.load(P + pool * RS + (start + c) * CS, live)
    tl.store(O + global_q, value.to(tl.int64), live)


@triton.jit
def _requests(
    I,  # noqa: E741
    P,
    S,
    O,  # noqa: E741
    N: tl.constexpr,
    D: tl.constexpr,
    IS: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    G: tl.constexpr,
):
    r = tl.program_id(0) * G + tl.arange(0, G)
    live = r < N
    pool = tl.load(I + r * IS, live).to(tl.int64)
    start = tl.load(S + r * SS, live).to(tl.int64)
    source = P + pool * RS + start * CS
    destination = O + r * D
    for col in range(D):
        value = tl.load(source + col * CS, live)
        tl.store(destination + col, value.to(tl.int64), live)


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
        if (
            req_to_token.stride(1) == 1
            and draft_token_num & (draft_token_num - 1) == 0
        ):
            cols = min(draft_token_num, 128)
            _fullrows_element[(triton.cdiv(total, 64 * cols),)](
                req_pool_indices,
                req_to_token,
                start_offset,
                out,
                batch_size,
                draft_token_num,
                *req_pool_indices.stride(),
                *req_to_token.stride(),
                *start_offset.stride(),
                C=cols,
                num_warps=4,
            )
        else:
            _requests[(triton.cdiv(batch_size, 64),)](
                req_pool_indices,
                req_to_token,
                start_offset,
                out,
                batch_size,
                draft_token_num,
                *req_pool_indices.stride(),
                *req_to_token.stride(),
                *start_offset.stride(),
                G=64,
                num_warps=4,
            )
    return out
