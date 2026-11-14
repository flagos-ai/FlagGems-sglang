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

"""Softcap logits in place on Enflame GCU (tuned on L600, kernel-mode do_bench).

y = cap * tanh(x / cap) is evaluated as (2*cap) * sigmoid(x * (2/cap)) - cap:
``tl.sigmoid`` lowers to the native single-op ``tops_sigmoid_v2_fp32`` while
``libdevice.tanh`` is an extern call that dominated the old kernel.  Measured
on 512x128256 fp32 (65.7M elems): old 625us -> 171us (3.7x); the pure
load+store floor on this board is ~172us, so the kernel is memory-bound.
Grid is a persistent 48-program cap (old cap was 12; even a pure copy is 1.4x
faster at 48 programs).  Dispatch, all measured:

    n >= 4M elems -> BLOCK=65536   (65.7M: 171us = 3.07TB/s)
    n >= 96K      -> BLOCK=8192    (262K: 4.7us, 2M: 10.8us)
    else          -> clamp(np2(n/16), 1024, 2048)  (4K: 3.3us, 32K: 3.1us)

``n_elements`` stays a constexpr: runtime-shaped block_ptr boundary checks
trip the ``memref_ext.slice_pad_start`` operandSegmentSizes bug, and the
module installs the same idempotent dialect shim as ``extend_attention.py``
so fresh compiles survive the 2026-11-07 toolchain regression.
"""

import re

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
    if op_name in ("memref.reinterpret_cast", "memref.subview") and not seg_match:
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = len([x for x in opers.group(1).split(",") if x.strip()]) if opers else 1
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
            content = "\n".join(_fix_gcu_dialect_line(l) for l in content.split("\n"))
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-11 重放剥离: 私有 GCUIR shim 剥尾零
# 误伤已修复工具链（fresh 编译实测 slice_pad_start 7 vs 8 全挂，同 bmm_chunk 重放
# 6 vs 7 实证）；现行策略 = 禁用全部 op 私有 shim（5dc28b9），剥离后 block_ptr
# constexpr 形态原生可编译。注释保留现场。

NUM_PROGRAMS = 48


@triton.jit
def _softcap_kernel(
    x_ptr,
    n_elements: tl.constexpr,
    two_inv_cap: tl.constexpr,
    two_cap: tl.constexpr,
    cap: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nblocks = tl.cdiv(n_elements, BLOCK)
    for bid in tl.range(pid, nblocks, nprog):
        blk = tl.make_block_ptr(
            x_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(bid * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        x = tl.load(blk, boundary_check=(0,)).to(tl.float32)
        y = two_cap * tl.sigmoid(x * two_inv_cap) - cap
        tl.store(blk, y.to(blk.dtype.element_ty), boundary_check=(0,))


def _pick_block(n):
    if n >= 4194304:
        return 65536
    if n >= 98304:
        return 8192
    return min(max(triton.next_power_of_2(max(n // 16, 1)), 1024), 2048)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    cap = float(final_logit_softcapping)
    x = (
        full_logits
        if full_logits.is_contiguous()
        else full_logits.contiguous()
    )
    n = int(x.numel())
    if n == 0:
        return full_logits
    block = _pick_block(n)
    nblocks = (n + block - 1) // block
    _softcap_kernel[(min(NUM_PROGRAMS, nblocks),)](
        x,
        n_elements=n,
        two_inv_cap=2.0 / cap,
        two_cap=2.0 * cap,
        cap=cap,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    if x is not full_logits:
        full_logits.copy_(x)
    return full_logits


__all__ = ["softcap_inplace_logits"]
