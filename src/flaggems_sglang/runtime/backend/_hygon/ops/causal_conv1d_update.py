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

import torch
import triton
import triton.language as tl

__all__ = ["causal_conv1d_update"]


_MAX_BLOCK = 256
_TARGET_PROGRAMS = 1024
_WARPS = 4


def _block_size(n_elements):
    block = min(triton.next_power_of_2(n_elements), _MAX_BLOCK)
    while block > 32 and triton.cdiv(n_elements, block) < _TARGET_PROGRAMS:
        block //= 2
    return block


@triton.jit
def _paired_kernel(
    x,
    state,
    weight,
    bias,
    out,
    new_state,
    n_pairs,
    dim: tl.constexpr,
    stride_bias: tl.constexpr,
    has_bias: tl.constexpr,
    use_silu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n_pairs
    d = i % (dim // 2)
    xp = tl.cast(x, tl.pointer_type(tl.uint32))
    sp = tl.cast(state, tl.pointer_type(tl.uint32))
    wp = tl.cast(weight, tl.pointer_type(tl.uint32))
    op = tl.cast(out, tl.pointer_type(tl.uint32))
    np = tl.cast(new_state, tl.pointer_type(tl.uint32))
    a = tl.load(sp + 3 * i, valid, other=0)
    b = tl.load(sp + 3 * i + 1, valid, other=0)
    c = tl.load(sp + 3 * i + 2, valid, other=0)
    v = tl.load(xp + i, valid, other=0)
    w0 = tl.load(wp + 4 * d, valid, other=0)
    w1 = tl.load(wp + 4 * d + 1, valid, other=0)
    w2 = tl.load(wp + 4 * d + 2, valid, other=0)
    w3 = tl.load(wp + 4 * d + 3, valid, other=0)
    dtype: tl.constexpr = x.dtype.element_ty
    s00 = (a & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    s01 = (a >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    s02 = (b & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    s10 = (b >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    s11 = (c & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    s12 = (c >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    x0 = (v & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    x1 = (v >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww0 = (w0 & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww1 = (w0 >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww2 = (w1 & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww3 = (w1 >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww4 = (w2 & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww5 = (w2 >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww6 = (w3 & 65535).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    ww7 = (w3 >> 16).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)
    y0 = s00 * ww0 + s01 * ww1 + s02 * ww2 + x0 * ww3
    y1 = s10 * ww4 + s11 * ww5 + s12 * ww6 + x1 * ww7
    if has_bias:
        y0 += tl.load(bias + 2 * d * stride_bias, valid, other=0).to(
            tl.float32
        )
        y1 += tl.load(bias + (2 * d + 1) * stride_bias, valid, other=0).to(
            tl.float32
        )
    if use_silu:
        y0 *= tl.sigmoid(y0)
        y1 *= tl.sigmoid(y1)
    o0 = y0.to(dtype).to(tl.uint16, bitcast=True).to(tl.uint32)
    o1 = y1.to(dtype).to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(op + i, o0 | (o1 << 16), valid)
    tl.store(np + 3 * i, (a >> 16) | (b << 16), valid)
    tl.store(np + 3 * i + 1, (v & 65535) | (c << 16), valid)
    tl.store(np + 3 * i + 2, (c >> 16) | (v & 4294901760), valid)


@triton.jit
def _single_token_kernel(
    x,
    state,
    weight,
    bias,
    out,
    new_state,
    n_elements,
    dim: tl.constexpr,
    state_len: tl.constexpr,
    width: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_st: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_od: tl.constexpr,
    stride_nb: tl.constexpr,
    stride_nd: tl.constexpr,
    stride_nt: tl.constexpr,
    stride_bias: tl.constexpr,
    has_bias: tl.constexpr,
    use_silu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    batch_idx = offsets // dim
    dim_idx = offsets - batch_idx * dim
    state_base = batch_idx * stride_sb + dim_idx * stride_sd
    weight_base = dim_idx * stride_wd
    x_value = tl.load(
        x + batch_idx * stride_xb + dim_idx * stride_xd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in range(width - 1):
        state_pos = tl.full((BLOCK,), state_len + 1 - width + k, tl.int32)
        weight_pos = tl.full((BLOCK,), k, tl.int32)
        state_value = tl.load(
            state + state_base + state_pos * stride_st,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight_value = tl.load(
            weight + weight_base + weight_pos * stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += state_value * weight_value
    last_pos = tl.full((BLOCK,), width - 1, tl.int32)
    last_weight = tl.load(
        weight + weight_base + last_pos * stride_wk,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc += x_value * last_weight
    if has_bias:
        acc += tl.load(bias + dim_idx * stride_bias, mask=mask, other=0.0).to(
            tl.float32
        )
    if use_silu:
        acc *= tl.sigmoid(acc)
    tl.store(
        out + batch_idx * stride_ob + dim_idx * stride_od,
        acc,
        mask=mask,
    )
    for pos in range(state_len - 1):
        source_pos = tl.full((BLOCK,), pos + 1, tl.int32)
        target_pos = tl.full((BLOCK,), pos, tl.int32)
        value = tl.load(
            state + state_base + source_pos * stride_st,
            mask=mask,
            other=0.0,
        )
        tl.store(
            new_state
            + batch_idx * stride_nb
            + dim_idx * stride_nd
            + target_pos * stride_nt,
            value,
            mask=mask,
        )
    final_pos = tl.full((BLOCK,), state_len - 1, tl.int32)
    tl.store(
        new_state
        + batch_idx * stride_nb
        + dim_idx * stride_nd
        + final_pos * stride_nt,
        x_value,
        mask=mask,
    )


@triton.jit
def _single_w4_s3_kernel(
    x,
    state,
    weight,
    bias,
    out,
    new_state,
    n_elements,
    dim: tl.constexpr,
    stride_bias: tl.constexpr,
    has_bias: tl.constexpr,
    use_silu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    dim_idx = offsets % dim
    state_base = offsets * 3
    weight_base = dim_idx * 4
    x_value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    s0 = tl.load(state + state_base, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(state + 1 + state_base, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(state + 2 + state_base, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight + weight_base, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight + 1 + weight_base, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight + 2 + weight_base, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight + 3 + weight_base, mask=mask, other=0.0).to(tl.float32)
    acc = s0 * w0 + s1 * w1 + s2 * w2 + x_value * w3
    if has_bias:
        acc += tl.load(bias + dim_idx * stride_bias, mask=mask, other=0.0).to(
            tl.float32
        )
    if use_silu:
        acc *= tl.sigmoid(acc)
    tl.store(out + offsets, acc, mask=mask)
    tl.store(new_state + state_base, s1, mask=mask)
    tl.store(new_state + 1 + state_base, s2, mask=mask)
    tl.store(new_state + 2 + state_base, x_value, mask=mask)


@triton.jit
def _output_kernel(
    x,
    state,
    weight,
    bias,
    out,
    n_elements,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    width: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_st: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_od: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_bias: tl.constexpr,
    has_bias: tl.constexpr,
    use_silu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    t = offsets % seqlen
    bd = offsets // seqlen
    batch_idx = bd // dim
    dim_idx = bd - batch_idx * dim
    state_base = batch_idx * stride_sb + dim_idx * stride_sd
    x_base = batch_idx * stride_xb + dim_idx * stride_xd
    weight_base = dim_idx * stride_wd
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in range(width):
        cat_offset = tl.full((BLOCK,), state_len + 1 - width + k, tl.int32)
        weight_pos = tl.full((BLOCK,), k, tl.int32)
        cat_idx = t + cat_offset
        from_state = cat_idx < state_len
        state_idx = tl.maximum(cat_idx, 0)
        x_idx = tl.maximum(cat_idx - state_len, 0)
        state_value = tl.load(
            state + state_base + state_idx * stride_st,
            mask=mask & from_state & (cat_idx >= 0),
            other=0.0,
        ).to(tl.float32)
        x_value = tl.load(
            x + x_base + x_idx * stride_xt,
            mask=mask
            & (~from_state)
            & (cat_idx >= state_len)
            & (x_idx < seqlen),
            other=0.0,
        ).to(tl.float32)
        weight_value = tl.load(
            weight + weight_base + weight_pos * stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += (state_value + x_value) * weight_value
    if has_bias:
        acc += tl.load(bias + dim_idx * stride_bias, mask=mask, other=0.0).to(
            tl.float32
        )
    if use_silu:
        acc *= tl.sigmoid(acc)
    tl.store(
        out + batch_idx * stride_ob + dim_idx * stride_od + t * stride_ot,
        acc,
        mask=mask,
    )


@triton.jit
def _state_kernel(
    x,
    state,
    new_state,
    n_elements,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_st: tl.constexpr,
    stride_nb: tl.constexpr,
    stride_nd: tl.constexpr,
    stride_nt: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    pos = offsets % state_len
    bd = offsets // state_len
    batch_idx = bd // dim
    dim_idx = bd - batch_idx * dim
    cat_idx = seqlen + pos
    from_state = cat_idx < state_len
    state_idx = tl.minimum(cat_idx, state_len - 1)
    x_idx = tl.maximum(cat_idx - state_len, 0)
    state_value = tl.load(
        state
        + batch_idx * stride_sb
        + dim_idx * stride_sd
        + state_idx * stride_st,
        mask=mask & from_state,
        other=0.0,
    )
    x_value = tl.load(
        x + batch_idx * stride_xb + dim_idx * stride_xd + x_idx * stride_xt,
        mask=mask & (~from_state) & (x_idx < seqlen),
        other=0.0,
    )
    tl.store(
        new_state
        + batch_idx * stride_nb
        + dim_idx * stride_nd
        + pos * stride_nt,
        tl.where(from_state, state_value, x_value),
        mask=mask,
    )


def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    is_2d = x.dim() == 2
    batch = x.shape[0]
    dim = x.shape[1]
    seqlen = 1 if is_2d else x.shape[2]
    state_len = conv_state.shape[2]
    width = weight.shape[1]
    out = torch.empty_like(x)
    new_conv_state = torch.empty_like(conv_state)
    if seqlen == 1:
        n_elements = batch * dim
        block = _block_size(n_elements)
        contiguous = (
            x.is_contiguous()
            and conv_state.is_contiguous()
            and weight.is_contiguous()
        )
        if (
            contiguous
            and width == 4
            and state_len == 3
            and dim % 2 == 0
            and x.dtype in (torch.float16, torch.bfloat16)
            and conv_state.dtype == x.dtype
            and weight.dtype == x.dtype
            and x.storage_offset() % 2 == 0
            and conv_state.storage_offset() % 2 == 0
            and weight.storage_offset() % 2 == 0
        ):
            _paired_kernel[(triton.cdiv(n_elements // 2, block),)](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                new_conv_state,
                n_elements // 2,
                dim=dim,
                stride_bias=0 if bias is None else bias.stride(0),
                has_bias=bias is not None,
                use_silu=activation in ("silu", "swish"),
                BLOCK=block,
                num_warps=_WARPS,
            )
        elif contiguous and width == 4 and state_len == 3:
            _single_w4_s3_kernel[(triton.cdiv(n_elements, block),)](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                new_conv_state,
                n_elements,
                dim=dim,
                stride_bias=0 if bias is None else bias.stride(0),
                has_bias=bias is not None,
                use_silu=activation in ("silu", "swish"),
                BLOCK=block,
                num_warps=_WARPS,
            )
        else:
            _single_token_kernel[(triton.cdiv(n_elements, block),)](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                new_conv_state,
                n_elements,
                dim=dim,
                state_len=state_len,
                width=width,
                stride_xb=x.stride(0),
                stride_xd=x.stride(1),
                stride_sb=conv_state.stride(0),
                stride_sd=conv_state.stride(1),
                stride_st=conv_state.stride(2),
                stride_wd=weight.stride(0),
                stride_wk=weight.stride(1),
                stride_ob=out.stride(0),
                stride_od=out.stride(1),
                stride_nb=new_conv_state.stride(0),
                stride_nd=new_conv_state.stride(1),
                stride_nt=new_conv_state.stride(2),
                stride_bias=0 if bias is None else bias.stride(0),
                has_bias=bias is not None,
                use_silu=activation in ("silu", "swish"),
                BLOCK=block,
                num_warps=_WARPS,
            )
    else:
        output_elements = batch * dim * seqlen
        output_block = _block_size(output_elements)
        _output_kernel[(triton.cdiv(output_elements, output_block),)](
            x,
            conv_state,
            weight,
            bias if bias is not None else weight,
            out,
            output_elements,
            dim=dim,
            seqlen=seqlen,
            state_len=state_len,
            width=width,
            stride_xb=x.stride(0),
            stride_xd=x.stride(1),
            stride_xt=x.stride(2),
            stride_sb=conv_state.stride(0),
            stride_sd=conv_state.stride(1),
            stride_st=conv_state.stride(2),
            stride_wd=weight.stride(0),
            stride_wk=weight.stride(1),
            stride_ob=out.stride(0),
            stride_od=out.stride(1),
            stride_ot=out.stride(2),
            stride_bias=0 if bias is None else bias.stride(0),
            has_bias=bias is not None,
            use_silu=activation in ("silu", "swish"),
            BLOCK=output_block,
            num_warps=_WARPS,
        )
        state_elements = batch * dim * state_len
        state_block = _block_size(state_elements)
        _state_kernel[(triton.cdiv(state_elements, state_block),)](
            x,
            conv_state,
            new_conv_state,
            state_elements,
            dim=dim,
            seqlen=seqlen,
            state_len=state_len,
            stride_xb=x.stride(0),
            stride_xd=x.stride(1),
            stride_xt=x.stride(2),
            stride_sb=conv_state.stride(0),
            stride_sd=conv_state.stride(1),
            stride_st=conv_state.stride(2),
            stride_nb=new_conv_state.stride(0),
            stride_nd=new_conv_state.stride(1),
            stride_nt=new_conv_state.stride(2),
            BLOCK=state_block,
            num_warps=_WARPS,
        )
    return out, new_conv_state
