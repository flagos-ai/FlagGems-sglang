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
_TARGET_PROGRAMS = 1
_WARPS = 4


def _block_size(n_elements):
    block = min(triton.next_power_of_2(n_elements), _MAX_BLOCK)
    while block > 32 and triton.cdiv(n_elements, block) < _TARGET_PROGRAMS:
        block //= 2
    return block


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
    dim: tl.constexpr,
    stride_bias: tl.constexpr,
    has_bias: tl.constexpr,
    use_silu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = dim_idx < dim
    offsets = batch_idx * dim + dim_idx
    x_value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    lane = tl.arange(0, BLOCK)
    linear = tl.arange(0, BLOCK * 4)
    base = tl.program_id(1) * BLOCK
    state_flat = tl.load(
        state + (batch_idx * dim + base) * 3 + linear,
        mask=(linear < BLOCK * 3) & (base * 3 + linear < dim * 3),
        other=0.0,
    ).to(tl.float32)
    weight_flat = tl.load(
        weight + base * 4 + linear, mask=base * 4 + linear < dim * 4, other=0.0
    ).to(tl.float32)
    s0 = tl.gather(state_flat, lane * 3, 0)
    s1 = tl.gather(state_flat, lane * 3 + 1, 0)
    s2 = tl.gather(state_flat, lane * 3 + 2, 0)
    w0 = tl.gather(weight_flat, lane * 4, 0)
    w1 = tl.gather(weight_flat, lane * 4 + 1, 0)
    w2 = tl.gather(weight_flat, lane * 4 + 2, 0)
    w3 = tl.gather(weight_flat, lane * 4 + 3, 0)
    acc = s0 * w0 + s1 * w1 + s2 * w2 + x_value * w3
    if has_bias:
        acc += tl.load(bias + dim_idx * stride_bias, mask=mask, other=0.0).to(
            tl.float32
        )
    if use_silu:
        acc *= tl.sigmoid(acc)
    tl.store(out + offsets, acc, mask=mask)
    shifted = tl.gather(state_flat, tl.minimum(linear + 1, BLOCK * 3 - 1), 0)
    x_gathered = tl.gather(x_value, tl.minimum(linear // 3, BLOCK - 1), 0)
    new_values = tl.where(linear % 3 == 2, x_gathered, shifted)
    tl.store(
        new_state + (batch_idx * dim + base) * 3 + linear,
        new_values,
        mask=(linear < BLOCK * 3) & (base * 3 + linear < dim * 3),
    )


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
            x.stride(0) == dim
            and conv_state.stride(0) == dim * state_len
            and out.stride(0) == dim
            and new_conv_state.stride(0) == dim * state_len
            and x.stride(1) == 1
            and conv_state.stride(2) == 1
            and conv_state.stride(1) == state_len
            and weight.stride(1) == 1
            and weight.stride(0) == width
            and out.stride(1) == 1
            and new_conv_state.stride(2) == 1
            and new_conv_state.stride(1) == state_len
        )
        if contiguous and width == 4 and state_len == 3:
            block = min(triton.next_power_of_2(dim), 512)
            _single_w4_s3_kernel[(batch, triton.cdiv(dim, block))](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                new_conv_state,
                dim=dim,
                stride_bias=bias.stride(0) if bias is not None else 1,
                has_bias=bias is not None,
                use_silu=activation in ("silu", "swish"),
                BLOCK=block,
                num_warps=_WARPS,
                multibuffer=False,
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
                stride_bias=bias.stride(0) if bias is not None else 1,
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
            stride_bias=bias.stride(0) if bias is not None else 1,
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
