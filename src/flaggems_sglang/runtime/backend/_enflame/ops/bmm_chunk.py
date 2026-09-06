# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_chunk_kernel(
    a,
    b,
    output,
    SEQLEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    K: tl.constexpr,
    NCHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    batch_chunk_group = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)
    group = batch_chunk_group % NGROUPS
    batch_chunk = batch_chunk_group // NGROUPS
    chunk = batch_chunk % NCHUNKS
    batch = batch_chunk // NCHUNKS
    rows_m = tile_m * 256 + tl.arange(0, 256)
    rows_n = tile_n * 256 + tl.arange(0, 256)
    offsets_k = tl.arange(0, 64)
    sequence_m = chunk * CHUNK_SIZE + rows_m
    sequence_n = chunk * CHUNK_SIZE + rows_n
    accumulator = tl.zeros((256, 256), tl.float32)

    for k_start in range(0, K, 64):
        current_k = k_start + offsets_k
        offsets_a = (
            (batch * SEQLEN + sequence_m[:, None]) * NGROUPS * K
            + group * K
            + current_k[None, :]
        )
        offsets_b = (
            (batch * SEQLEN + sequence_n[None, :]) * NGROUPS * K
            + group * K
            + current_k[:, None]
        )
        mask_a = (rows_m[:, None] < CHUNK_SIZE) & (current_k[None, :] < K)
        mask_b = (current_k[:, None] < K) & (rows_n[None, :] < CHUNK_SIZE)
        values_a = tl.load(a + offsets_a, mask=mask_a, other=0.0).to(
            tl.float32
        )
        values_b = tl.load(b + offsets_b, mask=mask_b, other=0.0).to(
            tl.float32
        )
        accumulator = tl.dot(values_a, values_b, accumulator)

    output_base = batch_chunk_group * CHUNK_SIZE * CHUNK_SIZE
    offsets_output = (
        output_base + rows_m[:, None] * CHUNK_SIZE + rows_n[None, :]
    )
    output_mask = (rows_m[:, None] < CHUNK_SIZE) & (
        rows_n[None, :] < CHUNK_SIZE
    )
    tl.store(output + offsets_output, accumulator, mask=output_mask)


def bmm_chunk(a, b, chunk_size, causal=False):
    batch, seqlen, ngroups, k = a.shape
    nchunks = seqlen // chunk_size
    output = torch.empty(
        (batch, nchunks, ngroups, chunk_size, chunk_size),
        device=a.device,
        dtype=a.dtype,
    )
    tiles = triton.cdiv(chunk_size, 256)
    _bmm_chunk_kernel[(batch * nchunks * ngroups, tiles, tiles)](
        a,
        b,
        output,
        SEQLEN=seqlen,
        NGROUPS=ngroups,
        K=k,
        NCHUNKS=nchunks,
        CHUNK_SIZE=chunk_size,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["bmm_chunk"]
