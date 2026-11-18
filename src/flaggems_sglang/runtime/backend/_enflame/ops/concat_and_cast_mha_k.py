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

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Toolchain-compat shim (2026-11-07 20:22 build skew): retry-path GCUIR
# downgrade scoped to this module's kernels. First compile of canonical 2D/3D
# tile loads (memref_ext.slice_*) fails on that build; the scoped retry
# rewrites the op encodings back to the old external-tool form. Fallback-only
# and name-scoped, so other kernels keep their exact compile path. Remove once
# the toolchain re-syncs (see compiler_issues/sglang_chunk_local_cumsum_vector.md).
# ---------------------------------------------------------------------------

_COMPAT_SCOPED_KERNELS = (
    "_concat_cast_mha_k_kernel",
    "_concat_cast_mha_k_kernel_rowloop",
    "_concat_cast_mha_k_kernel_flat",
)


def _downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""
    import re

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                line = re.sub(
                    r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                    r"operandSegmentSizes = array<i32: \1, 0>",
                )
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
                line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _install_gcuir_text_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_concat_cast_mha_k_compat", False):
        return
    original = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return original(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(name in text for name in _COMPAT_SCOPED_KERNELS):
                raise first_error
            try:
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._concat_cast_mha_k_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_gcuir_text_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。


@triton.jit
def _concat_cast_mha_k_kernel(
    N,
    R,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    TILES: tl.constexpr,
    GROUPS: tl.constexpr,
    FULL: tl.constexpr,
):
    nc = tl.arange(0, BN)
    rc = tl.arange(0, BR)
    for tile in tl.range(tl.program_id(0), TILES, tl.num_programs(0)):
        if BH == 1:
            token = (tile // GROUPS) * BT + tl.arange(0, BT)
            head0 = tile % GROUPS
            if FULL:
                nv = tl.load(
                    N + token[:, None] * NS0 + head0 * NS1 + nc[None, :] * NS2
                )
                rv = tl.load(R + token[:, None] * RS0 + rc[None, :] * RS2)
            else:
                nv = tl.load(
                    N + token[:, None] * NS0 + head0 * NS1 + nc[None, :] * NS2,
                    (token[:, None] < T) & (nc[None, :] < DN),
                    other=0,
                )
                rv = tl.load(
                    R + token[:, None] * RS0 + rc[None, :] * RS2,
                    (token[:, None] < T) & (rc[None, :] < DR),
                    other=0,
                )
            if HEAD_MAJOR:
                base = OUT + token[:, None] * H * (DN + DR) + head0
                if FULL:
                    tl.store(base + nc[None, :] * H, nv)
                    tl.store(base + (DN + rc[None, :]) * H, rv)
                else:
                    tl.store(
                        base + nc[None, :] * H,
                        nv,
                        (token[:, None] < T) & (nc[None, :] < DN),
                    )
                    tl.store(
                        base + (DN + rc[None, :]) * H,
                        rv,
                        (token[:, None] < T) & (rc[None, :] < DR),
                    )
            else:
                base = OUT + (token[:, None] * H + head0) * (DN + DR)
                if FULL:
                    tl.store(base + nc[None, :], nv)
                    tl.store(base + DN + rc[None, :], rv)
                else:
                    tl.store(
                        base + nc[None, :],
                        nv,
                        (token[:, None] < T) & (nc[None, :] < DN),
                    )
                    tl.store(
                        base + DN + rc[None, :],
                        rv,
                        (token[:, None] < T) & (rc[None, :] < DR),
                    )
        else:
            token = (tile // GROUPS) * BT + tl.arange(0, BT)
            head = (tile % GROUPS) * BH + tl.arange(0, BH)
            if FULL:
                nv = tl.load(
                    N
                    + token[:, None, None] * NS0
                    + head[None, :, None] * NS1
                    + nc[None, None, :] * NS2
                )
                rv = tl.load(R + token[:, None] * RS0 + rc[None, :] * RS2)
            else:
                nv = tl.load(
                    N
                    + token[:, None, None] * NS0
                    + head[None, :, None] * NS1
                    + nc[None, None, :] * NS2,
                    (token[:, None, None] < T)
                    & (head[None, :, None] < H)
                    & (nc[None, None, :] < DN),
                    other=0,
                )
                rv = tl.load(
                    R + token[:, None] * RS0 + rc[None, :] * RS2,
                    (token[:, None] < T) & (rc[None, :] < DR),
                    other=0,
                )
            if HEAD_MAJOR:
                base = (
                    OUT
                    + token[:, None, None] * H * (DN + DR)
                    + head[None, None, :]
                )
                if FULL:
                    tl.store(base + nc[None, :, None] * H, nv.trans(0, 2, 1))
                    tl.store(
                        base + (DN + rc[None, :, None]) * H,
                        tl.broadcast_to(rv[:, :, None], (BT, BR, BH)),
                    )
                else:
                    tl.store(
                        base + nc[None, :, None] * H,
                        nv.trans(0, 2, 1),
                        (token[:, None, None] < T)
                        & (head[None, None, :] < H)
                        & (nc[None, :, None] < DN),
                    )
                    tl.store(
                        base + (DN + rc[None, :, None]) * H,
                        tl.broadcast_to(rv[:, :, None], (BT, BR, BH)),
                        (token[:, None, None] < T)
                        & (head[None, None, :] < H)
                        & (rc[None, :, None] < DR),
                    )
            else:
                base = OUT + (
                    token[:, None, None] * H + head[None, :, None]
                ) * (DN + DR)
                if FULL:
                    tl.store(base + nc[None, None, :], nv)
                    tl.store(
                        base + DN + rc[None, None, :],
                        tl.broadcast_to(rv[:, None, :], (BT, BH, BR)),
                    )
                else:
                    tl.store(
                        base + nc[None, None, :],
                        nv,
                        (token[:, None, None] < T)
                        & (head[None, :, None] < H)
                        & (nc[None, None, :] < DN),
                    )
                    tl.store(
                        base + DN + rc[None, None, :],
                        tl.broadcast_to(rv[:, None, :], (BT, BH, BR)),
                        (token[:, None, None] < T)
                        & (head[None, :, None] < H)
                        & (rc[None, None, :] < DR),
                    )


@triton.jit
def _concat_cast_mha_k_kernel_flat(
    N,
    R,
    OUT,
    SIZE: tl.constexpr,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    PACK: tl.constexpr,
    SEL: tl.constexpr,
    B: tl.constexpr,
):
    if PACK:
        N = N.to(tl.pointer_type(tl.uint64))
        R = R.to(tl.pointer_type(tl.uint64))
        OUT = OUT.to(tl.pointer_type(tl.uint64))
    for tile in range(tl.program_id(0), tl.cdiv(SIZE, B), tl.num_programs(0)):
        i = tile * B + tl.arange(0, B)
        col = i % (DN + DR)
        row = i // (DN + DR)
        token = row // H
        head = row % H
        if SEL:
            ptr = tl.where(
                col < DN,
                N + token * NS0 + head * NS1 + col * NS2,
                R + token * RS0 + (col - DN) * RS2,
            )
            tl.store(OUT + i, tl.load(ptr, i < SIZE), i < SIZE)
        else:
            n = tl.load(
                N + token * NS0 + head * NS1 + col * NS2,
                (i < SIZE) & (col < DN),
                other=0,
            )
            r = tl.load(
                R + token * RS0 + (col - DN) * RS2,
                (i < SIZE) & (col >= DN),
                other=0,
            )
            tl.store(OUT + i, tl.where(col < DN, n, r), i < SIZE)


@triton.jit
def _concat_cast_mha_k_kernel_rowloop(
    N,
    R,
    OUT,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    B: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    row0 = tl.program_id(0) * B
    nc = tl.arange(0, BN)
    rc = tl.arange(0, BR)
    for i in range(B):
        row = row0 + i
        tok = row // H
        hd = row % H
        inb = row < ROWS
        nv = tl.load(
            N + tok * NS0 + hd * NS1 + nc * NS2,
            inb & (nc < DN),
            other=0,
        )
        rv = tl.load(
            R + tok * RS0 + rc * RS2,
            inb & (rc < DR),
            other=0,
        )
        ob = OUT + row * (DN + DR)
        tl.store(ob + nc, nv, inb & (nc < DN))
        tl.store(ob + DN + rc, rv, inb & (rc < DR))


def concat_and_cast_mha_k(k, k_nope, k_rope):
    t, h, dn = k_nope.shape
    dr = k_rope.shape[2]
    rows = t * h
    head_major = rows * (dn + dr) * k.element_size() >= 16 * 1024 * 1024
    if head_major:
        out = torch.empty_strided(
            (t, h, dn + dr),
            (h * (dn + dr), 1, h),
            dtype=k.dtype,
            device=k.device,
        )
    else:
        out = torch.empty((t, h, dn + dr), dtype=k.dtype, device=k.device)
    if rows * (dn + dr):
        bn = triton.next_power_of_2(dn)
        br = triton.next_power_of_2(dr)
        packable = (
            not head_major
            and k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
            and dn % 4 == dr % 4 == 0
            and k_nope.stride(2) == k_rope.stride(2) == 1
            and k_nope.stride(0) % 4 == k_nope.stride(1) % 4 == 0
            and k_rope.stride(0) % 4 == 0
            and k_nope.storage_offset() % 4 == k_rope.storage_offset() % 4 == 0
        )
        flat_block = None
        flat_grid = 0
        flat_warps = 1
        flat_sel = False
        if packable:
            size = rows * (dn + dr) // 4
            if rows <= 256:
                flat_block = 512
                flat_grid = triton.cdiv(size, 512)
                flat_warps = 4
                flat_sel = True
            else:
                block = triton.next_power_of_2(max(1, -(-size // 24)))
                if rows <= 4096 and block <= 8192:
                    flat_block = block
                    flat_grid = 24
                    flat_warps = 4
                    flat_sel = False
        if flat_block is not None:
            _concat_cast_mha_k_kernel_flat[(flat_grid,)](
                k_nope,
                k_rope,
                out,
                size,
                h,
                dn // 4,
                dr // 4,
                k_nope.stride(0) // 4,
                k_nope.stride(1) // 4,
                k_nope.stride(2),
                k_rope.stride(0) // 4,
                k_rope.stride(2),
                True,
                flat_sel,
                flat_block,
                num_warps=flat_warps,
            )
            return out
        if rows <= 64 and not head_major:
            _concat_cast_mha_k_kernel_rowloop[(rows,)](
                k_nope,
                k_rope,
                out,
                rows,
                h,
                dn,
                dr,
                k_nope.stride(0),
                k_nope.stride(1),
                k_nope.stride(2),
                k_rope.stride(0),
                k_rope.stride(2),
                1,
                bn,
                br,
                num_warps=1,
            )
            return out
        element_bytes = max(
            k.element_size(), k_nope.element_size(), k_rope.element_size()
        )
        bh = min(triton.next_power_of_2(h), 32, max(1, 16384 // max(bn, br)))
        token_limit = max(1, 524288 // (bh * max(bn, br) * element_bytes))
        bt = min(triton.next_power_of_2(t), 16, token_limit)
        tiles = triton.cdiv(t, bt) * triton.cdiv(h, bh)
        target = min(32, max(8, -(-rows // 512)))
        while tiles < target and (bt > 1 or bh > 1):
            if bt >= bh and bt > 1:
                bt //= 2
            elif bh > 1:
                bh //= 2
            else:
                break
            tiles = triton.cdiv(t, bt) * triton.cdiv(h, bh)
        groups = (h + bh - 1) // bh
        full = (t % bt == 0) and (h % bh == 0) and (bn == dn) and (br == dr)
        _concat_cast_mha_k_kernel[(min(48, tiles),)](
            k_nope,
            k_rope,
            out,
            t,
            h,
            dn,
            dr,
            k_nope.stride(0),
            k_nope.stride(1),
            k_nope.stride(2),
            k_rope.stride(0),
            k_rope.stride(2),
            bt,
            bh,
            bn,
            br,
            head_major,
            tiles,
            groups,
            full,
            num_warps=1,
        )
    return out


__all__ = ["concat_and_cast_mha_k"]
