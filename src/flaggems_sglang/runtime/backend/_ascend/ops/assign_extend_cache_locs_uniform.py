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

# Uniform cache gather semantics: SGLang (Apache-2.0), commit c07883d.
import torch
import triton
import triton.language as tl
from triton.runtime import driver

__all__ = ["assign_extend_cache_locs_uniform"]


@triton.jit
def _uniform(
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
):
    B: tl.constexpr = 256
    lane = tl.arange(0, B)
    for tile in range(tl.program_id(0), tl.cdiv(N * D, B), tl.num_programs(0)):
        q = tl.cast(tile, tl.int64) * B + lane
        row = q // D
        col = q % D
        live = q < N * D
        pool = tl.load(I + row * IS, live, 0).to(tl.int64)
        start = tl.load(S + row * SS, live, 0).to(tl.int64)
        value = tl.load(P + pool * RS + (start + col) * CS, live)
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
        cores = driver.active.utils.get_device_properties(
            req_to_token.device.index
        )["num_vectorcore"]
        _uniform[(min(triton.cdiv(total, 256), cores),)](
            req_pool_indices,
            req_to_token,
            start_offset,
            out,
            batch_size,
            draft_token_num,
            *req_pool_indices.stride(),
            *req_to_token.stride(),
            *start_offset.stride(),
            num_warps=4,
        )
    return out
