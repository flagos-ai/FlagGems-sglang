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


@triton.jit(do_not_specialize=["n"])
def _gather_kernel(
    idx_ptr,  # [n] int64 行索引
    p_ptr,  # [pool, W_P] topk_p 源
    ix_ptr,  # [pool, W_IX] topk_index 源
    tok_ptr,  # [pool] output_tokens 源
    hid_ptr,  # [pool, H] hidden_states 源 (H=0 时占位, 不解引用)
    p_out_ptr,  # [n, W_P]
    ix_out_ptr,  # [n, W_IX]
    tok_out_ptr,  # [n]
    hid_out_ptr,  # [n, H] (H=0 时占位, 不写入)
    n,
    R: tl.constexpr,  # 每 program 拷贝的行数
    W_P: tl.constexpr,
    W_IX: tl.constexpr,
    H: tl.constexpr,  # 0 表示无 hidden
):
    pid = tl.program_id(0)
    offs_w = tl.arange(0, 8)
    offs_h = tl.arange(0, H)
    for i in tl.static_range(R):
        row = pid * R + i
        if row < n:
            # 索引收窄到 int32: m*w 远小于 2^31, 减少地址折叠开销.
            src = tl.load(idx_ptr + row).to(tl.int32)
            v = tl.load(p_ptr + src * W_P + offs_w)
            tl.store(p_out_ptr + row * W_P + offs_w, v)
            v = tl.load(ix_ptr + src * W_IX + offs_w)
            tl.store(ix_out_ptr + row * W_IX + offs_w, v)
            v = tl.load(tok_ptr + src)
            tl.store(tok_out_ptr + row, v)
            if H > 0:
                v = tl.load(hid_ptr + src * H + offs_h)
                tl.store(hid_out_ptr + row * H + offs_h, v)


@triton.jit(do_not_specialize=["n"])
def _gather_kernel_generic(
    idx_ptr,
    p_ptr,
    ix_ptr,
    tok_ptr,
    hid_ptr,
    p_out_ptr,
    ix_out_ptr,
    tok_out_ptr,
    hid_out_ptr,
    n,
    R: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    W_P: tl.constexpr,
    W_IX: tl.constexpr,
    W_TOK: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_w = tl.arange(0, BLOCK_W)
    for i in tl.static_range(R):
        row = pid * R + i
        if row < n:
            src = tl.load(idx_ptr + row).to(tl.int32)
            if W_P == BLOCK_W:
                v = tl.load(p_ptr + src * W_P + offs_w)
                tl.store(p_out_ptr + row * W_P + offs_w, v)
            else:
                mask = offs_w < W_P
                v = tl.load(p_ptr + src * W_P + offs_w, mask=mask)
                tl.store(p_out_ptr + row * W_P + offs_w, v, mask=mask)
            if W_IX == BLOCK_W:
                v = tl.load(ix_ptr + src * W_IX + offs_w)
                tl.store(ix_out_ptr + row * W_IX + offs_w, v)
            else:
                mask = offs_w < W_IX
                v = tl.load(ix_ptr + src * W_IX + offs_w, mask=mask)
                tl.store(ix_out_ptr + row * W_IX + offs_w, v, mask=mask)
            if W_TOK == BLOCK_W:
                v = tl.load(tok_ptr + src * W_TOK + offs_w)
                tl.store(tok_out_ptr + row * W_TOK + offs_w, v)
            elif W_TOK == 1:
                v = tl.load(tok_ptr + src)
                tl.store(tok_out_ptr + row, v)
            else:
                mask = offs_w < W_TOK
                v = tl.load(tok_ptr + src * W_TOK + offs_w, mask=mask)
                tl.store(tok_out_ptr + row * W_TOK + offs_w, v, mask=mask)
            if H > 0:
                offs_h = tl.arange(0, BLOCK_H)
                if H == BLOCK_H:
                    v = tl.load(hid_ptr + src * H + offs_h)
                    tl.store(hid_out_ptr + row * H + offs_h, v)
                else:
                    mask = offs_h < H
                    v = tl.load(hid_ptr + src * H + offs_h, mask=mask)
                    tl.store(hid_out_ptr + row * H + offs_h, v, mask=mask)


_INT64 = torch.int64  # 常量绑定: 避免热路径上的属性查找.
_NEW_EMPTY = torch.Tensor.new_empty

_MAX_GRID = 65535


def _next_pow2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _pick_rw(m):
    """(R, num_warps) 按 m 分档. 本机 A/B 实测选点, 无全局可变状态."""
    if m < 8:
        return 1, 2
    if m < 64:
        return 3, 2
    if m < 512:
        return 4, 1
    if m < 2048:
        return 6, 2
    return 4, 2


def gather_spec_extras(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    m = indices.shape[0]
    if m == 0:
        return (
            topk_p_buf[:0],
            topk_index_buf[:0],
            output_tokens_buf[:0],
            (hidden_states_buf[:0] if hidden_states_buf is not None else None),
        )

    if (
        indices.dtype is _INT64
        and hidden_states_buf is not None
        and indices.is_contiguous()
        and topk_p_buf.is_contiguous()
        and topk_index_buf.is_contiguous()
        and output_tokens_buf.is_contiguous()
        and hidden_states_buf.is_contiguous()
        and topk_p_buf.dim() == 2
        and topk_index_buf.dim() == 2
        and output_tokens_buf.dim() == 1
        and hidden_states_buf.dim() == 2
    ):
        w_p = topk_p_buf.shape[1]
        w_ix = topk_index_buf.shape[1]
        if w_p <= 8 and w_ix <= 8:
            h = hidden_states_buf.shape[1]
            # 输出用模板 new_empty: 继承 dtype/device, 省掉 device-guard 与内
            # 部格式协商 (本机后端实测比 torch.empty(shape, dtype=, device=) 快).
            out_p = _NEW_EMPTY(topk_p_buf, (m, w_p))
            out_ix = _NEW_EMPTY(topk_index_buf, (m, w_ix))
            out_tok = _NEW_EMPTY(output_tokens_buf, (m,))
            out_hid = _NEW_EMPTY(hidden_states_buf, (m, h))
            r, nw = _pick_rw(m)
            grid = (-(-m // r),)
            while grid[0] > _MAX_GRID:
                r *= 2
                grid = (-(-m // r),)
            _gather_kernel[grid](
                indices,
                topk_p_buf,
                topk_index_buf,
                output_tokens_buf,
                hidden_states_buf,
                out_p,
                out_ix,
                out_tok,
                out_hid,
                m,
                R=r,
                W_P=w_p,
                W_IX=w_ix,
                H=h,
                num_warps=nw,
            )
            return (out_p, out_ix, out_tok, out_hid)

    return _generic(
        indices,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
        m,
    )


def _generic(indices, p_src, ix_src, tok_src, hid_src, m):
    """任意 dtype/布局/None-hidden 的通用路径 (标准 launch)."""
    idx = indices.contiguous()
    if idx.dtype is not _INT64:
        idx = idx.to(_INT64)
    p = p_src.contiguous()
    ix = ix_src.contiguous()
    tok = tok_src.contiguous()
    p2 = p.dim() > 1
    ix2 = ix.dim() > 1
    tok2 = tok.dim() > 1
    w_p = p.shape[1] if p2 else 1
    w_ix = ix.shape[1] if ix2 else 1
    w_tok = tok.shape[1] if tok2 else 1
    ne = _NEW_EMPTY
    out_p = ne(p, (m, w_p)) if p2 else ne(p, (m,))
    out_ix = ne(ix, (m, w_ix)) if ix2 else ne(ix, (m,))
    out_tok = ne(tok, (m, w_tok)) if tok2 else ne(tok, (m,))

    if hid_src is not None:
        hid = hid_src.contiguous()
        h = hid.shape[1]
        out_hid = ne(hid, (m, h))
    else:
        # 占位指针, H=0 的特化不会解引用它.
        hid = p
        h = 0
        out_hid = None

    r = _pick_rw(m)[0]
    grid = (-(-m // r),)
    _gather_kernel_generic[grid](
        idx,
        p,
        ix,
        tok,
        hid,
        out_p,
        out_ix,
        out_tok,
        out_hid if out_hid is not None else p,
        m,
        R=r,
        BLOCK_W=_next_pow2(max(w_p, w_ix, w_tok)),
        BLOCK_H=_next_pow2(h),
        W_P=w_p,
        W_IX=w_ix,
        W_TOK=w_tok,
        H=h,
        num_warps=_pick_rw(m)[1],
    )
    return (out_p, out_ix, out_tok, out_hid)


__all__ = ["gather_spec_extras"]
