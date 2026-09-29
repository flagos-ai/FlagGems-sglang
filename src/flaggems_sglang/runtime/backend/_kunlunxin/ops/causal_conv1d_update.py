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


_MAX_BLOCK = 4096
_TARGET_PROGRAMS = 1
_WARPS = 1


def _block_size(n_elements):
    block = min(triton.next_power_of_2(n_elements), _MAX_BLOCK)
    while block > 32 and triton.cdiv(n_elements, block) < _TARGET_PROGRAMS:
        block //= 2
    return block


@triton.jit
def _single_batch_kernel(
    x,
    state,
    weight,
    bias,
    out,
    new_state,
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
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dim_idx < dim
    state_base = batch_idx * stride_sb + dim_idx * stride_sd
    weight_base = dim_idx * stride_wd
    x_value = tl.load(
        x + batch_idx * stride_xb + dim_idx * stride_xd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for k in range(width - 1):
        state_pos = tl.full((BLOCK_D,), state_len + 1 - width + k, tl.int32)
        weight_pos = tl.full((BLOCK_D,), k, tl.int32)
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
    last_pos = tl.full((BLOCK_D,), width - 1, tl.int32)
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
        source_pos = tl.full((BLOCK_D,), pos + 1, tl.int32)
        target_pos = tl.full((BLOCK_D,), pos, tl.int32)
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
    final_pos = tl.full((BLOCK_D,), state_len - 1, tl.int32)
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


@triton.jit
def _state_s3_flat_kernel(
    x,
    state,
    new_state,
    n_state_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_state_elements
    pos = offsets % 3
    source_offsets = tl.minimum(offsets + 1, n_state_elements - 1)
    state_value = tl.load(state + source_offsets, mask=mask, other=0.0)
    x_value = tl.load(x + offsets // 3, mask=mask, other=0.0)
    tl.store(
        new_state + offsets,
        tl.where(pos < 2, state_value, x_value),
        mask=mask,
    )


@triton.jit
def _output_position_kernel(
    x,
    state,
    weight,
    bias,
    out,
    n_elements,
    dim: tl.constexpr,
    position: tl.constexpr,
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
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    mask = offsets < n_elements
    batch_idx = (offsets // dim).to(tl.int32)
    dim_idx = (offsets - batch_idx * dim).to(tl.int32)
    state_base = (batch_idx * stride_sb + dim_idx * stride_sd).to(tl.int32)
    x_base = (batch_idx * stride_xb + dim_idx * stride_xd).to(tl.int32)
    weight_base = (dim_idx * stride_wd).to(tl.int32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in range(width):
        cat_position = position + state_len + 1 - width + k
        weight_pos = tl.full((BLOCK,), k, tl.int32)
        if cat_position < state_len:
            source_pos = tl.full((BLOCK,), cat_position, tl.int32)
            value = tl.load(
                state + state_base + source_pos * stride_st,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        else:
            source_pos = tl.full((BLOCK,), cat_position - state_len, tl.int32)
            value = tl.load(
                x + x_base + source_pos * stride_xt,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        weight_value = tl.load(
            weight + weight_base + weight_pos * stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += value * weight_value
    if has_bias:
        acc += tl.load(bias + dim_idx * stride_bias, mask=mask, other=0.0).to(
            tl.float32
        )
    if use_silu:
        acc *= tl.sigmoid(acc)
    output_pos = tl.full((BLOCK,), position, tl.int32)
    tl.store(
        out
        + batch_idx * stride_ob
        + dim_idx * stride_od
        + output_pos * stride_ot,
        acc,
        mask=mask,
    )


@triton.jit
def _copy_position_kernel(
    source,
    new_state,
    n_elements,
    dim: tl.constexpr,
    source_pos: tl.constexpr,
    target_pos: tl.constexpr,
    stride_src_b: tl.constexpr,
    stride_src_d: tl.constexpr,
    stride_src_t: tl.constexpr,
    stride_nb: tl.constexpr,
    stride_nd: tl.constexpr,
    stride_nt: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    mask = offsets < n_elements
    batch_idx = (offsets // dim).to(tl.int32)
    dim_idx = (offsets - batch_idx * dim).to(tl.int32)
    source_position = tl.full((BLOCK,), source_pos, tl.int32)
    target_position = tl.full((BLOCK,), target_pos, tl.int32)
    value = tl.load(
        source
        + batch_idx * stride_src_b
        + dim_idx * stride_src_d
        + source_position * stride_src_t,
        mask=mask,
        other=0.0,
    )
    tl.store(
        new_state
        + batch_idx * stride_nb
        + dim_idx * stride_nd
        + target_position * stride_nt,
        value,
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
        contiguous = (
            x.is_contiguous()
            and conv_state.is_contiguous()
            and weight.is_contiguous()
        )
        if (
            contiguous
            and width == 4
            and state_len == 3
            and dim % 16 == 0
            and x.storage_offset() % 16 == 0
            and conv_state.storage_offset() % 16 == 0
            and weight.storage_offset() % 16 == 0
        ):
            n_elements = batch * dim
            block = _block_size(n_elements)
            _single_w4_s3_kernel[(triton.cdiv(n_elements, block),)](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                n_elements,
                dim=dim,
                stride_bias=0 if bias is None else bias.stride(0),
                has_bias=bias is not None,
                use_silu=activation in ("silu", "swish"),
                BLOCK=block,
                num_warps=_WARPS,
            )
            n_state_elements = n_elements * 3
            state_block = min(triton.next_power_of_2(n_state_elements), 8192)
            _state_s3_flat_kernel[
                (triton.cdiv(n_state_elements, state_block),)
            ](
                x,
                conv_state,
                new_conv_state,
                n_state_elements,
                BLOCK=state_block,
                num_warps=_WARPS,
            )
        else:
            block_d = min(triton.next_power_of_2(dim), 512)
            _single_batch_kernel[(batch, triton.cdiv(dim, block_d))](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                new_conv_state,
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
                BLOCK_D=block_d,
                num_warps=_WARPS,
            )
    else:
        n_elements = batch * dim
        block = _block_size(n_elements)
        grid = (triton.cdiv(n_elements, block),)
        for position in range(seqlen):
            _output_position_kernel[grid](
                x,
                conv_state,
                weight,
                bias if bias is not None else weight,
                out,
                n_elements,
                dim=dim,
                position=position,
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
                BLOCK=block,
                num_warps=1,
            )
        for target_pos in range(state_len):
            cat_pos = seqlen + target_pos
            if cat_pos < state_len:
                source = conv_state
                source_pos = cat_pos
                source_strides = conv_state.stride()
            else:
                source = x
                source_pos = cat_pos - state_len
                source_strides = x.stride()
            _copy_position_kernel[grid](
                source,
                new_conv_state,
                n_elements,
                dim=dim,
                source_pos=source_pos,
                target_pos=target_pos,
                stride_src_b=source_strides[0],
                stride_src_d=source_strides[1],
                stride_src_t=source_strides[2],
                stride_nb=new_conv_state.stride(0),
                stride_nd=new_conv_state.stride(1),
                stride_nt=new_conv_state.stride(2),
                BLOCK=block,
                num_warps=1,
            )
    return out, new_conv_state
