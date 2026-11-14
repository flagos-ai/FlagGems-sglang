# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""extend_attention for the Enflame GCU (single-kernel direct-gather rewrite).

Baseline note (2026-11-07 sweep, enflame 1.80/36.07/31.08/52.92 ms core) ran on
old-stack cached binaries: on the 2026-11-07 20:22 toolchain the previous
two-pass implementation (gather kernel + fp32-IEEE dot main kernel, nw=4,
BLOCK_M=16) no longer first-compiles (memref_ext.slice_start operandSegmentSizes
7 vs 6, see compiler_issues/sglang_extend_attention.md) and was 2.0-2.7x
slower than the generic implementation besides.

This rewrite ports the delivered vllm triton_unified_attention recipe (same
paged-GQA flash online-softmax family, job j20261108105833_831e6050):

1. GCUIR dialect shim (process-global, idempotent) wrapping
   ``triton.backends.enflame.toolkit.gcu_compiler_opt`` so rank-2 / masked
   access and ``gcu.matmul`` compile against the older external
   gcu-compiler-opt (same shim as flaggems_vllm ops/kda64.py, verified there
   in production for this exact kernel family).
2. Single kernel: the separate gather pass is removed; K/V are gathered
   directly from the token-indexed KV buffer per tile, through a same-width
   ``torch.uint16`` view bit-cast in-kernel (bf16 gather tiles feeding tl.dot
   lower through a slow dot-consumer path on this backend; the integer channel
   is bit-identical). Non-contiguous buffers or non-power-of-2 head dims keep
   the direct-load arm.
3. GQA packing: BLOCK_M packs all query heads of one KV head
   (rows = BLOCK_Q_POS positions x group_size heads), cutting K/V reads by
   group_size; BLOCK_M = next_pow2(group * min(32, max_extend)) capped 128,
   TILE_KV = 64, num_warps = 1, num_stages = 2 (swept: TILE 32 +7-25%,
   BLOCK_M 64 +56-94%, num_stages 3 flat -- all landed on the vllm values).
4. bf16/fp16 MMA dots with fp32 accumulators (the old fp32-IEEE casts were
   both slow and information-free on 2-byte inputs); standard online softmax
   with a single final normalization instead of per-tile divisions.
5. Host: no ``.item()`` device sync (uses the caller-provided
   max_len_extend), int32 index normalization with an int64 arm for
   >2^31-element KV buffers.

Correctness hardening, compile-time gated to ``head_dim != next_pow2(head_dim)``
(D=80-style paths; all bench/production shapes take the fast path untouched):
on the 20:22 toolchain, masked out-of-bounds lanes on that path were observed
to read pool garbage (0 x Inf -> NaN at the last token row / last KV group,
pool-state dependent; generic implementation shows the identical latent NaN)
and dead-row stores to be re-targeted. The gated branch clamps every address
into the valid region, value-masks loads, and stores dead rows to a dummy row
(``total_extend + 1`` allocation, zero-copy view returned). Unclamped affine
addressing on the fast path is deliberate: a ``tl.minimum`` clamp turns the
affine Q/K/V tiles into data-dependent gathers and costs 2x (measured).

Measured (GCU400, official bench core, kernel mode, 3-run mean, spread<=0.1%):
B1 0.309 ms (2.69x generic, 5.83x old enflame), B8 6.282 ms (2.87x / 5.74x),
B64 5.792 ms (2.03x / 5.37x), B256 8.068 ms (2.39x / 6.56x);
pytest --quick 7/7; dual-path accuracy: enflame max_abs vs fp32 reference
<= 3.99e-3 on all 7 cases, within 1e-2 gate, near bit-identical to generic
(<= 4.8e-7) on the 6 shared-clean cases.

uint16-view loads keep TILE_KV <= 64 (a 128-wide masked u16 load row trips an
ld.lld relocation failure, see compiler_issues/ld_lld_relocation.md), and the
u16 arm additionally requires head_dim == padded head_dim so the dim mask is
all-true and folded (no 128-lane non-uniform bool on the load).
"""

import math
import re

import torch
import triton
import triton.language as tl

_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if (
        op_name in ("memref.reinterpret_cast", "memref.subview")
        and not seg_match
    ):
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = (
            len([x for x in opers.group(1).split(",") if x.strip()])
            if opers
            else 1
        )
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(
                _fix_gcu_dialect_line(line) for line in content.split("\n")
            )
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


def _downgrade_gcuir_for_external_tool(text):
    out_lines = []
    for line in text.splitlines():
        line = _fix_gcu_dialect_line(line)
        if '"gcu.matmul"' in line:
            line = re.sub(
                r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                r"operandSegmentSizes = array<i32: \1, 0>",
                line,
            )
            line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
            line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _patch_gcu_matmul_compat():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(
        _toolkit.gcu_compiler_opt, "_flaggems_kda_matmul_compat", False
    ):
        return
    _prev = _toolkit.gcu_compiler_opt

    def _compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if '"gcu.matmul"' not in text or "gcu.matmul" not in str(
                first_error
            ):
                raise first_error
            try:
                return _toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    *args,
                )
            except Exception:
                raise first_error

    _compat._flaggems_kda_matmul_compat = True
    _toolkit.gcu_compiler_opt = _compat


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6
# _patch_gcu_matmul_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6

_MAX_PROGRAMS = 8192


@triton.jit
def _dot_qk(q, k):
    if q.dtype == tl.float32 and k.dtype == tl.float32:
        return tl.dot(q, k, out_dtype=tl.float32, input_precision="ieee")
    else:
        return tl.dot(q, k, out_dtype=tl.float32)


@triton.jit
def _dot_pv(p, v):
    if p.dtype == tl.float32 and v.dtype == tl.float32:
        return tl.dot(p, v, out_dtype=tl.float32, input_precision="ieee")
    else:
        return tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)


@triton.jit
def _extend_attention_kernel(
    q_ptr,
    ke_ptr,
    ve_ptr,
    kb_ptr,
    vb_ptr,
    out_ptr,
    qo_ptr,
    kvp_ptr,
    kvi_ptr,
    sm_scale,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    buffer_tokens,
    total_extend,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_ke_token,
    stride_ke_head,
    stride_ke_dim,
    stride_ve_token,
    stride_ve_head,
    stride_ve_dim,
    stride_kb_token,
    stride_kb_head,
    stride_kb_dim,
    stride_vb_token,
    stride_vb_head,
    stride_vb_dim,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    pid_base,
    num_query_blocks,
    block_m: tl.constexpr,
    block_q_pos: tl.constexpr,
    tile_kv: tl.constexpr,
    block_d: tl.constexpr,
    kv_u16: tl.constexpr,
    idx_i64: tl.constexpr,
):
    task = pid_base + tl.program_id(0)
    query_block = task % num_query_blocks
    head_task = task // num_query_blocks
    kv_head_id = head_task % num_kv_heads
    sequence_id = head_task // num_kv_heads

    query_start = tl.load(qo_ptr + sequence_id)
    query_end = tl.load(qo_ptr + sequence_id + 1)
    extend_length = query_end - query_start
    q_block_start = query_block * block_q_pos
    if q_block_start >= extend_length:
        return
    prefix_start = tl.load(kvp_ptr + sequence_id)
    prefix_end = tl.load(kvp_ptr + sequence_id + 1)
    prefix_length = prefix_end - prefix_start

    offs_m = tl.arange(0, block_m)
    q_pos_local = offs_m // group_size
    head_in_group = offs_m % group_size
    query_pos = q_block_start + q_pos_local
    query_head = kv_head_id * group_size + head_in_group
    q_mask = (q_pos_local < block_q_pos) & (query_pos < extend_length)

    offs_d = tl.arange(0, block_d)
    d_mask = offs_d < head_dim

    # Defensive hardening, compile-time gated to non-power-of-2 head dims
    # (see module docstring / compiler issue sglang_extend_attention.md):
    # on that path masked out-of-bounds lanes were observed to read garbage
    # (0 x Inf -> NaN) and dead-row stores to be re-targeted, so addresses
    # are clamped into the valid region and loads are value-masked. On the
    # power-of-2 path (all bench/production shapes) every mask is provably
    # honored, and clamping would turn the affine Q/K/V tiles into
    # data-dependent gathers (measured 2x slower), so the fast affine
    # lowering is kept untouched.
    if head_dim != block_d:
        q_pos_load = tl.minimum(query_pos, extend_length - 1)
    else:
        q_pos_load = query_pos
    q = tl.load(
        q_ptr
        + (query_start + q_pos_load)[:, None] * stride_q_token
        + query_head[:, None] * stride_q_head
        + offs_d[None, :] * stride_q_dim,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    if head_dim != block_d:
        q = tl.where(d_mask[None, :], q, 0.0)

    m_i = tl.full([block_m], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([block_m], dtype=tl.float32)
    acc = tl.zeros([block_m, block_d], dtype=tl.float32)

    offs_t = tl.arange(0, tile_kv)

    # Phase 1: prefix keys gathered from the token-indexed KV buffer.
    for key_start in range(0, prefix_length, tile_kv):
        key_offsets = key_start + offs_t
        key_valid = key_offsets < prefix_length
        if head_dim != block_d:
            key_offsets_load = tl.minimum(key_offsets, prefix_length - 1)
        else:
            key_offsets_load = key_offsets
        token_ids = tl.load(
            kvi_ptr + prefix_start + key_offsets_load, mask=key_valid, other=0
        )
        token_ids = tl.where(
            token_ids < 0, token_ids + buffer_tokens, token_ids
        )
        if head_dim != block_d:
            token_ids = tl.minimum(tl.maximum(token_ids, 0), buffer_tokens - 1)
        if idx_i64:
            token_ids = token_ids.to(tl.int64)
        k_ptrs = (
            kb_ptr
            + token_ids[None, :] * stride_kb_token
            + kv_head_id * stride_kb_head
            + offs_d[:, None] * stride_kb_dim
        )
        v_ptrs = (
            vb_ptr
            + token_ids[:, None] * stride_vb_token
            + kv_head_id * stride_vb_head
            + offs_d[None, :] * stride_vb_dim
        )
        if kv_u16:
            k = tl.load(
                k_ptrs, mask=d_mask[:, None] & key_valid[None, :], other=0
            ).to(q_ptr.dtype.element_ty, bitcast=True)
            v = tl.load(
                v_ptrs, mask=key_valid[:, None] & d_mask[None, :], other=0
            ).to(q_ptr.dtype.element_ty, bitcast=True)
        else:
            k = tl.load(
                k_ptrs, mask=d_mask[:, None] & key_valid[None, :], other=0.0
            )
            v = tl.load(
                v_ptrs, mask=key_valid[:, None] & d_mask[None, :], other=0.0
            )
        if head_dim != block_d:
            k = tl.where(d_mask[:, None], k, 0.0)
            v = tl.where(d_mask[None, :], v, 0.0)
        s = _dot_qk(q, k) * sm_scale
        s = tl.where(key_valid[None, :] & q_mask[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + _dot_pv(p, v)
        m_i = m_safe

    # Phase 2: extend keys, causal, affine loads from the contiguous tensors.
    extend_limit = tl.minimum(q_block_start + block_q_pos, extend_length)
    for key_start in range(0, extend_limit, tile_kv):
        key_offsets = key_start + offs_t
        key_valid = key_offsets < extend_length
        if head_dim != block_d:
            key_offsets_load = tl.minimum(key_offsets, extend_length - 1)
        else:
            key_offsets_load = key_offsets
        k = tl.load(
            ke_ptr
            + (query_start + key_offsets_load)[None, :] * stride_ke_token
            + kv_head_id * stride_ke_head
            + offs_d[:, None] * stride_ke_dim,
            mask=d_mask[:, None] & key_valid[None, :],
            other=0.0,
        )
        v = tl.load(
            ve_ptr
            + (query_start + key_offsets_load)[:, None] * stride_ve_token
            + kv_head_id * stride_ve_head
            + offs_d[None, :] * stride_ve_dim,
            mask=key_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if head_dim != block_d:
            k = tl.where(d_mask[:, None], k, 0.0)
            v = tl.where(d_mask[None, :], v, 0.0)
        s = _dot_qk(q, k) * sm_scale
        causal = key_offsets[None, :] <= query_pos[:, None]
        s = tl.where(
            causal & key_valid[None, :] & q_mask[:, None], s, float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + _dot_pv(p, v)
        m_i = m_safe

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    # Dead rows store to the dummy row (total_extend) on the non-power-of-2
    # path so a store-mask miss lands in scratch, never on real rows; on the
    # power-of-2 path the store mask is honored and the affine address is
    # kept for the fast lowering.
    if head_dim != block_d:
        store_token = tl.where(q_mask, query_start + query_pos, total_extend)
    else:
        store_token = query_start + query_pos
    tl.store(
        out_ptr
        + store_token[:, None] * stride_out_token
        + query_head[:, None] * stride_out_head
        + offs_d[None, :] * stride_out_dim,
        out.to(out_ptr.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )


def _prepare_index(tensor, device):
    return tensor.to(device=device, dtype=torch.int32).contiguous()


def extend_attention(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
):
    total_extend, num_heads, head_dim = q_extend.shape
    num_kv_heads = k_extend.shape[1]
    group_size = max(1, num_heads // num_kv_heads)
    batch_size = qo_indptr.numel() - 1
    if batch_size <= 0 or total_extend == 0:
        return torch.zeros_like(q_extend, dtype=torch.float32)
    device = q_extend.device
    # One dummy row absorbs any store-mask miss from dead rows (see module
    # docstring); the returned tensor is the zero-copy view of the real rows.
    output = torch.empty(
        (total_extend + 1, num_heads, head_dim),
        dtype=torch.float32,
        device=device,
    )
    result = output[:total_extend]
    qo = _prepare_index(qo_indptr, device)
    kvp = _prepare_index(kv_indptr, device)
    kvi = _prepare_index(kv_indices, device)
    max_length = max(int(max_len_extend), 1)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_m = max(
        16,
        min(128, triton.next_power_of_2(group_size * min(32, max_length))),
    )
    if block_m < group_size:
        block_m = triton.next_power_of_2(group_size)
    block_q_pos = max(1, block_m // group_size)
    tile_kv = 64

    use_u16 = (
        q_extend.dtype == k_buffer.dtype == v_buffer.dtype
        and k_buffer.dtype in (torch.bfloat16, torch.float16)
        and k_buffer.is_contiguous()
        and v_buffer.is_contiguous()
        and head_dim == block_d
    )
    kb_arg = k_buffer.view(torch.uint16) if use_u16 else k_buffer
    vb_arg = v_buffer.view(torch.uint16) if use_u16 else v_buffer
    idx_i64 = k_buffer.numel() >= 2**31 or v_buffer.numel() >= 2**31

    num_query_blocks = triton.cdiv(max_length, block_q_pos)
    total_programs = num_query_blocks * num_kv_heads * batch_size
    for pid_base in range(0, total_programs, _MAX_PROGRAMS):
        launch_programs = min(_MAX_PROGRAMS, total_programs - pid_base)
        _extend_attention_kernel[launch_programs,](
            q_extend,
            k_extend,
            v_extend,
            kb_arg,
            vb_arg,
            output,
            qo,
            kvp,
            kvi,
            1.0 / math.sqrt(head_dim),
            num_kv_heads,
            group_size,
            head_dim,
            k_buffer.shape[0],
            total_extend,
            q_extend.stride(0),
            q_extend.stride(1),
            q_extend.stride(2),
            k_extend.stride(0),
            k_extend.stride(1),
            k_extend.stride(2),
            v_extend.stride(0),
            v_extend.stride(1),
            v_extend.stride(2),
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            v_buffer.stride(0),
            v_buffer.stride(1),
            v_buffer.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            pid_base,
            num_query_blocks,
            block_m=block_m,
            block_q_pos=block_q_pos,
            tile_kv=tile_kv,
            block_d=block_d,
            kv_u16=use_u16,
            idx_i64=idx_i64,
            num_warps=1,
            num_stages=2,
        )
    return result


__all__ = ["extend_attention"]
