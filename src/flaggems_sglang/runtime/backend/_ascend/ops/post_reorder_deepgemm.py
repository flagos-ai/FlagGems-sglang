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

"""post_reorder_deepgemm: MoE DeepGEMM epilogue 的纯 Triton 实现."""

import triton
import triton.language as tl


@triton.jit
def _post_reorder_2d_kernel(
    down_ptr,  # (num_tokens * topk, hidden) down proj 输出
    out_ptr,  # (num_tokens, hidden)
    src2dst_ptr,  # (num_tokens, topk) int32
    ids_ptr,  # (num_tokens, topk) int32, -1 = pad
    w_ptr,  # (num_tokens, topk) float32
    H,  # hidden_size (行 stride, 输入连续)
    scaling,
    TOPK: tl.constexpr,  # 实际 topk
    INDEX64: tl.constexpr,  # True 时用 int64 偏移 (张量超过 2^31 元素)
    BLOCK_H: tl.constexpr,  # 每个 program 覆盖的列数 (2 的幂)
):
    t = tl.program_id(0)
    hb = tl.program_id(1)

    h_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        id_k = tl.load(ids_ptr + t * TOPK + k)
        dst_k = tl.maximum(tl.load(src2dst_ptr + t * TOPK + k), 0)
        w_k = tl.load(w_ptr + t * TOPK + k).to(tl.float32)
        # 无效 slot 权重置 0 + scaling 融进权重 (消掉最后的向量 mul)
        w_eff = tl.where(id_k >= 0, w_k, 0.0) * scaling
        if INDEX64:
            row = tl.load(
                down_ptr + dst_k.to(tl.int64) * H + h_offs,
                mask=h_mask,
                other=0.0,
            )
        else:
            row = tl.load(
                down_ptr + dst_k * H + h_offs,
                mask=h_mask,
                other=0.0,
            )
        acc += row.to(tl.float32) * w_eff

    tl.store(
        out_ptr + t * H + h_offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=h_mask,
    )


@triton.jit
def _post_reorder_serial_kernel(
    down_ptr,  # (num_tokens * topk, hidden) down proj 输出
    out_ptr,  # (num_tokens, hidden)
    src2dst_ptr,  # (num_tokens, topk) int32
    ids_ptr,  # (num_tokens, topk) int32, -1 = pad
    w_ptr,  # (num_tokens, topk) float32
    H,  # hidden_size (行 stride, 输入连续)
    NCH,  # hidden 分块数 = cdiv(H, CH)
    scaling,
    TOPK: tl.constexpr,  # 实际 topk
    INDEX64: tl.constexpr,  # True 时用 int64 偏移 (张量超过 2^31 元素)
    CH: tl.constexpr,  # 每个 chunk 的列数 (2 的幂)
):
    t = tl.program_id(0)

    for c in range(NCH):
        h_offs = c * CH + tl.arange(0, CH)
        h_mask = h_offs < H

        acc = tl.zeros([CH], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            id_k = tl.load(ids_ptr + t * TOPK + k)
            dst_k = tl.maximum(tl.load(src2dst_ptr + t * TOPK + k), 0)
            w_k = tl.load(w_ptr + t * TOPK + k).to(tl.float32)
            w_eff = tl.where(id_k >= 0, w_k, 0.0) * scaling
            if INDEX64:
                row = tl.load(
                    down_ptr + dst_k.to(tl.int64) * H + h_offs,
                    mask=h_mask,
                    other=0.0,
                )
            else:
                row = tl.load(
                    down_ptr + dst_k * H + h_offs,
                    mask=h_mask,
                    other=0.0,
                )
            acc += row.to(tl.float32) * w_eff

        tl.store(
            out_ptr + t * H + h_offs,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
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
    if num_tokens == 0:
        return output

    # 偏移量最大值 = down_output.numel() <= 2^31-1 时 int32 足够
    index64 = down_output.numel() > (2**31 - 1)

    # 宿主端静态配置 (本机 Ascend 910 实测最优, 详见文件头注释)
    scaling = float(routed_scaling_factor)
    if num_tokens <= 64:
        block_h = 4096
        num_warps = 2
        grid = (num_tokens, triton.cdiv(hidden_size, block_h))
        _post_reorder_2d_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            hidden_size,
            scaling,
            TOPK=topk,
            INDEX64=index64,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    else:
        # 大 M: 串行 chunk 循环, program 数 = M, 降派发成本.
        # CH=1024: H=7168 恰为 7*1024, 无尾部掩码浪费 (CH=2048 会浪费 12.5%).
        chunk = 1024
        grid = (num_tokens,)
        _post_reorder_serial_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            hidden_size,
            triton.cdiv(hidden_size, chunk),
            scaling,
            TOPK=topk,
            INDEX64=index64,
            CH=chunk,
            num_warps=1,
        )
    return output


__all__ = ["post_reorder_deepgemm"]
