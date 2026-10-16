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

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # latency-bound small token counts: one wide H tile per token
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 256}, num_warps=2, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 256}, num_warps=1, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=2, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=1, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=1, num_stages=4
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=1, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 512}, num_warps=1, num_stages=1
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=8, num_stages=5
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=16, num_stages=5
        ),
        triton.Config(
            {"BLOCK_T": 2, "BLOCK_H": 512}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 2, "BLOCK_H": 1024}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 2, "BLOCK_H": 512}, num_warps=8, num_stages=2
        ),
        # mid token counts: ramp toward full streaming bandwidth
        triton.Config(
            {"BLOCK_T": 4, "BLOCK_H": 512}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 4, "BLOCK_H": 512}, num_warps=16, num_stages=5
        ),
        triton.Config(
            {"BLOCK_T": 4, "BLOCK_H": 1024}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 8, "BLOCK_H": 256}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_T": 8, "BLOCK_H": 256}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 8, "BLOCK_H": 512}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 8, "BLOCK_H": 512}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_T": 8, "BLOCK_H": 1024}, num_warps=8, num_stages=2
        ),
        # large token counts: saturate HBM streaming
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 256}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 256}, num_warps=8, num_stages=5
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 512}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 512}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_H": 256}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_H": 512}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_H": 512}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 64, "BLOCK_H": 256}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 64, "BLOCK_H": 512}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 128, "BLOCK_H": 256}, num_warps=8, num_stages=3
        ),
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def _post_reorder_deepgemm_kernel(
    down_ptr,  # (num_tokens * topk, hidden_size) gathered expert outputs
    out_ptr,  # (num_tokens, hidden_size)
    src2dst_ptr,  # (num_tokens, topk) int32 destination rows
    ids_ptr,  # (num_tokens, topk) int32 expert ids, -1 = padding
    w_ptr,  # (num_tokens, topk) float32 routing weights
    num_tokens,
    hidden_size,
    scaling,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_T: tl.constexpr,  # num_tokens % BLOCK_T == 0: skip token mask
    EVEN_H: tl.constexpr,  # hidden_size % BLOCK_H == 0: skip hidden mask
    WIDE: tl.constexpr,  # emit 64-bit offsets (needed only past int32 range)
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_h = tl.multiple_of(offs_h, 16)
    if EVEN_T:
        t_mask = tl.full((BLOCK_T,), 1, tl.int1)
    else:
        t_mask = offs_t < num_tokens
    if EVEN_H:
        h_mask = tl.full((BLOCK_H,), 1, tl.int1)
    else:
        h_mask = offs_h < hidden_size
    mask2d = t_mask[:, None] & h_mask[None, :]

    if WIDE:
        out_offs = offs_t[:, None].to(tl.int64) * hidden_size + offs_h[None, :]
    else:
        out_offs = offs_t[:, None] * hidden_size + offs_h[None, :]

    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        slot = offs_t * TOPK + k
        ids_k = tl.load(ids_ptr + slot, mask=t_mask, other=-1)
        valid_k = ids_k >= 0
        w_k = tl.load(w_ptr + slot, mask=t_mask & valid_k, other=0.0)
        # Invalid slots contribute exactly zero; keep their row pointer at 0.
        dst_k = tl.load(src2dst_ptr + slot, mask=t_mask & valid_k, other=0)
        if WIDE:
            row_base = dst_k.to(tl.int64) * hidden_size
        else:
            row_base = dst_k * hidden_size
        if EVEN_T:
            rows = tl.load(
                down_ptr + row_base[:, None] + offs_h[None, :],
                mask=valid_k[:, None] & h_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            rows = tl.load(
                down_ptr + row_base[:, None] + offs_h[None, :],
                mask=mask2d & valid_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
        acc += rows.to(tl.float32) * w_k[:, None]

    acc = acc * scaling
    tl.store(
        out_ptr + out_offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask2d,
        eviction_policy="evict_first",
    )


def post_reorder_deepgemm(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    # 64-bit offsets are only needed when a linear index can exceed int32.
    max_elem = down_output.numel() + num_tokens * hidden_size
    wide = max_elem > 2**31 - 1

    def grid(meta):
        return (
            triton.cdiv(num_tokens, meta["BLOCK_T"]),
            triton.cdiv(hidden_size, meta["BLOCK_H"]),
        )

    _post_reorder_deepgemm_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        num_tokens,
        hidden_size,
        routed_scaling_factor,
        TOPK=topk,
        # Every BLOCK_T in the config space is a power of two <= 128, so
        # num_tokens % 128 == 0 implies num_tokens % BLOCK_T == 0.
        EVEN_T=(num_tokens % 128 == 0),
        # Every BLOCK_H in the config space divides 1024, so
        # hidden % 1024 == 0 implies hidden % BLOCK_H == 0.
        EVEN_H=(hidden_size % 1024 == 0),
        WIDE=wide,
    )
    return output


__all__ = ["post_reorder_deepgemm"]
