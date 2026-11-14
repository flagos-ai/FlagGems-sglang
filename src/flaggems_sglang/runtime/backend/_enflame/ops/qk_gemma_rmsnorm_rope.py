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

# GCU notes: the upstream 2D (heads x dim) tile kernels hit the
# memref_ext.slice_start/slice_pad_start operandSegmentSizes miscompile
# family on the 2026-11-07 toolchain (fresh compile fails; see
# compiler_issues/sglang_2d_tile_fresh_compile_slice_pad_start.md), so
# this copy is rewritten as strictly-1D kernels mirroring the delivered
# fused_qk_gemma_rmsnorm_rope_gate rewrite:
#   * row-stream arm (general fallback): persistent grid-stride +
#     static_range rows, strictly 1D affine per-row accesses;
#   * token arms (common full-RoPE neox bf16 contiguous layout): groups
#     aligned to tokens so cos/sin are shared affine [HALF] loads and the
#     position is one scalar per token -- the data-dependent per-row
#     cos/sin gathers of the row-batched form measured ~6x costlier than
#     everything else in this op (85% of kernel time at t=512).  Small t
#     uses flat (token, head-block) groups for machine fill; large t uses
#     one fused Q+K group per token.
#
# Round 2 (2026-11-13, E-1 second pass, GCU400): ablation at t=512
# attributed the remaining time to (a) the per-token pos->cos/sin
# dependent narrow-load chain (~32us exposed) and (b) the per-row
# rmsnorm reduction trees (~26us).  Two bit-exact structural changes
# landed (noop floor 2.5us, L0 copy floor 25.1us stay untouched):
#   * tokf arm reorder: the four payload loads are issued before the
#     pos/cos/sin loads (fast path when NQ==RQ and NK==RK), so the
#     tree+norm math covers the cos/sin latency (74.1 vs 83.9us @t=512,
#     18.2 vs 19.3us @t=64).
#   * tokm arm (t>=256, even, single Q/K head-block): adjacent token
#     pairs' Q groups are merged into one (2,RQ,HALF) group -- one
#     4096-lane load, one 3D tree, one set of rope vector ops and two flat
#     stores instead of two of each; the K side merges four tokens
#     ((4,RK,HALF), 4*RK<=32 rows) into a (KM,1,HALF) cos/sin gather over
#     one 128/256-lane load.  Halving the per-token vector-op count
#     measured 66.0 vs 83.9us @t=512 (-21.5%) and 423 vs 611us @t=4096
#     (-30.8%), bit-exact.  Merged-group row count stays <=64
#     (2*RQ<=64, KM*RK<=32): (128,64) reshapes measured
#     silent-wrong-values + 3.2x slower (merge4 probe), matching the
#     fused_qk_gemma_rmsnorm_rope_gate R>=64 precedent.  Odd t, t<256
#     and multi head-block head counts (nq>32-pow2 families) fall back
#     to the reordered tokf arm at unchanged speed.

import torch
import triton
import triton.language as tl


def _install_qk_gemma_gcuir_compat():
    """Scoped GCUIR compat for this kernel family.

    The process-global dialect shims installed by other enflame ops
    (extend_attention on gcu_compiler_opt, decode_grouped_attention on
    toolkit._run_command) rewrite operandSegmentSizes attributes on
    ``memref_ext.slice_*`` lines; combined with the current external
    gcu-compiler-opt this can fail first-compiles with "size mismatch in
    attribute conversion" (see
    compiler_issues/sglang_l2norm_masked2d_slice_pad_start.md and
    compiler_issues/sglang_2d_tile_fresh_compile_slice_pad_start.md).
    Retry those compiles (and only those, matched on this op's kernel
    symbol prefix) by invoking the external tool directly via subprocess
    -- both wrapped layers would re-strip the retry IR (qkv_lora_b
    pattern). Fallback-only: every already-compilable kernel keeps its
    exact code path.
    """
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_qk_gemma_gcuir_compat", False):
        return
    import subprocess

    _prev = toolkit.gcu_compiler_opt

    def _raw_gcu_compiler_opt(text, *args):
        tool = toolkit._require_toolkit_tool("gcu-compiler-opt")
        result = subprocess.run(
            [tool, "-mlir-print-op-generic", *args],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        return result.stdout

    def gcu_compiler_opt_compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if "@_fused_qk_gemma" not in text:
                raise first_error
            try:
                return _raw_gcu_compiler_opt(text, *args)
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._qk_gemma_gcuir_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_qk_gemma_gcuir_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。


# Rows per static_range group in the row-stream arm: dispatch cost
# amortization vs tail waste.
_ROWS = 8
# Persistent grid cap (multiples of the 48 SPCs).
_GRID_TARGET = 48
_WARPS = 1
# Token-arm dispatch line: at >= this many tokens the fused per-token
# groups (shared pos/cos/sin for Q+K, no loop-carried small loads) beat
# the flat (token, head-block) groups; below it the flat form fills more
# SPCs.
_TOKEN_FUSED_MIN_T = 48
# Merged-pair arm dispatch line: at >= this many tokens (and even t,
# single Q/K head-block) the two-token merged Q groups amortize their
# wider ops past the tokf arm; below it the two grid-stride loops
# underfill the 48 SPCs (t=128: 30.4 vs 25.1us, t=192: 33.1 vs 32.2us,
# crossover measured between 192 and 256; t=256: 41.8 vs 42.6us).
_TOKEN_MERGE_MIN_T = 256


@triton.jit
def _gemma_rope_row(
    X,
    W,
    C,
    P,
    Y,
    row,
    w1,
    w2,
    wt,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    NEOX: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
):
    # One (token, head) row, strictly 1D affine accesses.
    token = row // H
    head = row % H
    xbase = token * XS0 + head * D * XS1
    out = row * D
    HALF: tl.constexpr = RD // 2
    p = tl.arange(0, BP)
    if NEOX:
        i1 = p
        i2 = HALF + p
    else:
        i1 = 2 * p
        i2 = 2 * p + 1
    EVENP: tl.constexpr = BP == HALF
    if EVENP:
        x1 = tl.load(X + xbase + i1 * XS1, cache_modifier=".cg").to(tl.float32)
        x2 = tl.load(X + xbase + i2 * XS1, cache_modifier=".cg").to(tl.float32)
        squares = tl.sum(x1 * x1 + x2 * x2, 0)
    else:
        pm = p < HALF
        x1 = tl.load(
            X + xbase + i1 * XS1, pm, 0, cache_modifier=".cg"
        ).to(tl.float32)
        x2 = tl.load(
            X + xbase + i2 * XS1, pm, 0, cache_modifier=".cg"
        ).to(tl.float32)
        squares = tl.sum(x1 * x1 + x2 * x2, 0)
    if RD < D:
        TAIL: tl.constexpr = D - RD
        t = RD + tl.arange(0, BT)
        if BT == TAIL:
            xt = tl.load(X + xbase + t * XS1, cache_modifier=".cg").to(
                tl.float32
            )
            squares += tl.sum(xt * xt, 0)
        else:
            tm = t < D
            xt = tl.load(
                X + xbase + t * XS1, tm, 0, cache_modifier=".cg"
            ).to(tl.float32)
            squares += tl.sum(xt * xt, 0)
    inv = tl.rsqrt(squares / D + EPS)
    if RD > 0:
        if EVENP:
            n1 = (x1 * inv * (w1 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            n2 = (x2 * inv * (w2 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + p * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + p) * CS1).to(tl.float32)
            tl.store(Y + out + i1, n1 * cos - n2 * sin)
            tl.store(Y + out + i2, n2 * cos + n1 * sin)
        else:
            pm = p < HALF
            n1 = (x1 * inv * (w1 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            n2 = (x2 * inv * (w2 + 1.0)).to(Y.dtype.element_ty).to(tl.float32)
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + p * CS1, pm, 0).to(tl.float32)
            sin = tl.load(
                C + pos * CS0 + (HALF + p) * CS1, pm, 0
            ).to(tl.float32)
            tl.store(Y + out + i1, n1 * cos - n2 * sin, pm)
            tl.store(Y + out + i2, n2 * cos + n1 * sin, pm)
    if RD < D:
        t = RD + tl.arange(0, BT)
        if BT == TAIL:
            nt = (xt * inv * (wt + 1.0)).to(Y.dtype.element_ty)
            tl.store(Y + out + t, nt)
        else:
            tm = t < D
            nt = (xt * inv * (wt + 1.0)).to(Y.dtype.element_ty)
            tl.store(Y + out + t, nt, tm)


@triton.jit
def _fused_qk_gemma_rmsnorm_rope_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    NQ_ROWS,
    NK_ROWS,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    NEOX: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Loop-invariant weights: loaded once per program instead of per row.
    HALF: tl.constexpr = RD // 2
    p = tl.arange(0, BP)
    if NEOX:
        iw1 = p
        iw2 = HALF + p
    else:
        iw1 = 2 * p
        iw2 = 2 * p + 1
    EVENP: tl.constexpr = BP == HALF
    if EVENP:
        qw1 = tl.load(QW + iw1 * QWS).to(tl.float32)
        qw2 = tl.load(QW + iw2 * QWS).to(tl.float32)
        kw1 = tl.load(KW + iw1 * KWS).to(tl.float32)
        kw2 = tl.load(KW + iw2 * KWS).to(tl.float32)
    else:
        wm = p < HALF
        qw1 = tl.load(QW + iw1 * QWS, wm, 0).to(tl.float32)
        qw2 = tl.load(QW + iw2 * QWS, wm, 0).to(tl.float32)
        kw1 = tl.load(KW + iw1 * KWS, wm, 0).to(tl.float32)
        kw2 = tl.load(KW + iw2 * KWS, wm, 0).to(tl.float32)
    qwt = 0.0
    kwt = 0.0
    if RD < D:
        TAIL: tl.constexpr = D - RD
        t = RD + tl.arange(0, BT)
        if BT == TAIL:
            qwt = tl.load(QW + t * QWS).to(tl.float32)
            kwt = tl.load(KW + t * KWS).to(tl.float32)
        else:
            tm = t < D
            qwt = tl.load(QW + t * QWS, tm, 0).to(tl.float32)
            kwt = tl.load(KW + t * KWS, tm, 0).to(tl.float32)
    # Persistent grid-stride over flat rows: [0, NQ_ROWS) are Q rows,
    # [NQ_ROWS, NQ_ROWS + NK_ROWS) are K rows.
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    total = NQ_ROWS + NK_ROWS
    for grp in tl.range(pid, tl.cdiv(total, ROWS), npid):
        base = grp * ROWS
        for r in tl.static_range(ROWS):
            row = base + r
            if row < NQ_ROWS:
                _gemma_rope_row(
                    Q, QW, C, P, QO, row, qw1, qw2, qwt,
                    NQ, D, RD, NEOX, EPS, QS0, QS1, CS0, CS1, PS, BP, BT,
                )
            elif row < total:
                _gemma_rope_row(
                    K, KW, C, P, KO, row - NQ_ROWS, kw1, kw2, kwt,
                    NK, D, RD, NEOX, EPS, KS0, KS1, CS0, CS1, PS, BP, BT,
                )


@triton.jit
def _rope_group(
    X,
    Y,
    token,
    h0,
    w1,
    w2,
    cos,
    sin,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
):
    # R heads of one token: 1D div/mod x loads with HALF-lane contiguous
    # segments, register reshape for the batched axis=1 tree, flat stores.
    j = tl.arange(0, R * HALF)
    h = j // HALF
    p = j % HALF
    base = token * XS0 + (h0 + h) * D + p
    x1 = tl.load(X + base, cache_modifier=".cg").to(tl.float32)
    x2 = tl.load(X + base + HALF, cache_modifier=".cg").to(tl.float32)
    x1r = tl.reshape(x1, (R, HALF))
    x2r = tl.reshape(x2, (R, HALF))
    squares = tl.sum(x1r * x1r + x2r * x2r, 1)
    inv = tl.rsqrt(squares / D + EPS)
    n1 = (
        (x1r * inv[:, None] * (w1[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    n2 = (
        (x2r * inv[:, None] * (w2[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    o1 = n1 * cos[None, :] - n2 * sin[None, :]
    o2 = n2 * cos[None, :] + n1 * sin[None, :]
    out = (token * H + h0 + h) * D + p
    tl.store(Y + out, tl.reshape(o1, (R * HALF,)), cache_modifier=".cg")
    tl.store(
        Y + out + HALF, tl.reshape(o2, (R * HALF,)), cache_modifier=".cg"
    )


@triton.jit
def _rope_load(
    X,
    token,
    h0,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    XS0: tl.constexpr,
):
    # Payload half-pair for one (token, h0:h0+R) group; issued separately
    # so the tokf fast path can overlap the narrow pos/cos/sin loads.
    j = tl.arange(0, R * HALF)
    h = j // HALF
    p = j % HALF
    base = token * XS0 + (h0 + h) * D + p
    x1 = tl.load(X + base, cache_modifier=".cg").to(tl.float32)
    x2 = tl.load(X + base + HALF, cache_modifier=".cg").to(tl.float32)
    return x1, x2


@triton.jit
def _rope_finish(
    X,
    Y,
    token,
    h0,
    w1,
    w2,
    cos,
    sin,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    x1,
    x2,
):
    # _rope_group with preloaded payload (bit-exact same math).
    x1r = tl.reshape(x1, (R, HALF))
    x2r = tl.reshape(x2, (R, HALF))
    squares = tl.sum(x1r * x1r + x2r * x2r, 1)
    inv = tl.rsqrt(squares / D + EPS)
    n1 = (
        (x1r * inv[:, None] * (w1[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    n2 = (
        (x2r * inv[:, None] * (w2[None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    o1 = n1 * cos[None, :] - n2 * sin[None, :]
    o2 = n2 * cos[None, :] + n1 * sin[None, :]
    j = tl.arange(0, R * HALF)
    h = j // HALF
    p = j % HALF
    out = (token * H + h0 + h) * D + p
    tl.store(Y + out, tl.reshape(o1, (R * HALF,)), cache_modifier=".cg")
    tl.store(
        Y + out + HALF, tl.reshape(o2, (R * HALF,)), cache_modifier=".cg"
    )


@triton.jit
def _rope_merge_group(
    X,
    Y,
    t0,
    h0,
    w1,
    w2,
    cosr,
    sinr,
    U: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
):
    # U adjacent tokens' groups merged into one (U, R, HALF) group: one
    # wide load per half, one 3D axis=2 tree, one set of rope vector ops,
    # two flat stores.  cosr/sinr are (U, 1, HALF) token-broadcast views.
    # Keep U*R <= 64: (128, 64)-class reshapes measured silent-wrong-value
    # + 3.2x slower (see header notes).
    j = tl.arange(0, U * R * HALF)
    jt = j // (R * HALF)
    jr = j % (R * HALF)
    h = jr // HALF
    p = jr % HALF
    base = (t0 + jt) * XS0 + (h0 + h) * D + p
    x1 = tl.load(X + base, cache_modifier=".cg").to(tl.float32)
    x2 = tl.load(X + base + HALF, cache_modifier=".cg").to(tl.float32)
    x1r = tl.reshape(x1, (U, R, HALF))
    x2r = tl.reshape(x2, (U, R, HALF))
    squares = tl.sum(x1r * x1r + x2r * x2r, 2)
    inv = tl.rsqrt(squares / D + EPS)
    n1 = (
        (x1r * inv[:, :, None] * (w1[None, None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    n2 = (
        (x2r * inv[:, :, None] * (w2[None, None, :] + 1.0))
        .to(Y.dtype.element_ty)
        .to(tl.float32)
    )
    o1 = n1 * cosr - n2 * sinr
    o2 = n2 * cosr + n1 * sinr
    out = ((t0 + jt) * H + h0 + h) * D + p
    tl.store(Y + out, tl.reshape(o1, (U * R * HALF,)), cache_modifier=".cg")
    tl.store(
        Y + out + HALF, tl.reshape(o2, (U * R * HALF,)), cache_modifier=".cg"
    )


@triton.jit
def _fused_qk_gemma_token_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    T,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    KS0: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
):
    # Flat (token, head-block) groups: [0, T*QHB) are Q groups, the rest
    # are K groups; host guarantees RQ | NQ and RK | NK so every group is
    # full. Weights are loop-invariant; pos/cos/sin are one scalar plus
    # two affine [HALF] loads per group.
    HALF: tl.constexpr = D // 2
    pw = tl.arange(0, HALF)
    qw1 = tl.load(QW + pw * QWS).to(tl.float32)
    qw2 = tl.load(QW + (HALF + pw) * QWS).to(tl.float32)
    kw1 = tl.load(KW + pw * KWS).to(tl.float32)
    kw2 = tl.load(KW + (HALF + pw) * KWS).to(tl.float32)
    QHB: tl.constexpr = NQ // RQ
    KHB: tl.constexpr = NK // RK
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    q_groups = T * QHB
    n_groups = q_groups + T * KHB
    for grp in tl.range(pid, n_groups, npid):
        if grp < q_groups:
            token = grp // QHB
            h0 = (grp % QHB) * RQ
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
            _rope_group(
                Q, QO, token, h0, qw1, qw2, cos, sin,
                RQ, NQ, D, HALF, EPS, QS0,
            )
        else:
            g2 = grp - q_groups
            token = g2 // KHB
            h0 = (g2 % KHB) * RK
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
            _rope_group(
                K, KO, token, h0, kw1, kw2, cos, sin,
                RK, NK, D, HALF, EPS, KS0,
            )


@triton.jit
def _fused_qk_gemma_tokf_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    T,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    KS0: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
):
    # Round-2 reorder: the payload loads are issued before the narrow
    # pos/cos/sin loads (fast path: one Q + one K head-block), so the
    # tree+norm math covers the pos->cos/sin dependent-load latency
    # instead of stalling on it (bit-exact; ~-12% @t=512, ~-5% @t=64).
    HALF: tl.constexpr = D // 2
    pw = tl.arange(0, HALF)
    qw1 = tl.load(QW + pw * QWS).to(tl.float32)
    qw2 = tl.load(QW + (HALF + pw) * QWS).to(tl.float32)
    kw1 = tl.load(KW + pw * KWS).to(tl.float32)
    kw2 = tl.load(KW + (HALF + pw) * KWS).to(tl.float32)
    QHB: tl.constexpr = NQ // RQ
    KHB: tl.constexpr = NK // RK
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    for token in tl.range(pid, T, npid):
        if (QHB == 1) and (KHB == 1):
            qx1, qx2 = _rope_load(Q, token, 0, RQ, NQ, D, HALF, QS0)
            kx1, kx2 = _rope_load(K, token, 0, RK, NK, D, HALF, KS0)
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
            _rope_finish(
                Q, QO, token, 0, qw1, qw2, cos, sin,
                RQ, NQ, D, HALF, EPS, QS0, qx1, qx2,
            )
            _rope_finish(
                K, KO, token, 0, kw1, kw2, cos, sin,
                RK, NK, D, HALF, EPS, KS0, kx1, kx2,
            )
        else:
            pos = tl.load(P + token * PS)
            cos = tl.load(C + pos * CS0 + pw * CS1).to(tl.float32)
            sin = tl.load(C + pos * CS0 + (HALF + pw) * CS1).to(tl.float32)
            for hb in tl.static_range(QHB):
                _rope_group(
                    Q, QO, token, hb * RQ, qw1, qw2, cos, sin,
                    RQ, NQ, D, HALF, EPS, QS0,
                )
            for hb in tl.static_range(KHB):
                _rope_group(
                    K, KO, token, hb * RK, kw1, kw2, cos, sin,
                    RK, NK, D, HALF, EPS, KS0,
                )


@triton.jit
def _fused_qk_gemma_tokm_kernel(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    T,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    KS0: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
    KM: tl.constexpr,
):
    # Merged arm (t >= _TOKEN_MERGE_MIN_T, even t): Q groups merge two
    # adjacent tokens ((2,RQ,HALF), 2*RQ <= 64 rows), K groups merge KM
    # tokens ((KM,RK,HALF), KM*RK <= 32; KM=4 when t%4==0 else 2).  The
    # cos/sin gather for a merged K group is one (KM*HALF)-lane load over
    # per-token position scalars; each merged group halves (or quarters)
    # the per-token vector-op count vs the tokf arm.  Host guarantees
    # t even and t % KM == 0 so every merged group is full.
    HALF: tl.constexpr = D // 2
    qw1 = tl.load(QW + tl.arange(0, HALF) * QWS).to(tl.float32)
    qw2 = tl.load(QW + (HALF + tl.arange(0, HALF)) * QWS).to(tl.float32)
    kw1 = tl.load(KW + tl.arange(0, HALF) * KWS).to(tl.float32)
    kw2 = tl.load(KW + (HALF + tl.arange(0, HALF)) * KWS).to(tl.float32)
    QHB: tl.constexpr = NQ // RQ
    KHB: tl.constexpr = NK // RK
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    cj = tl.arange(0, 2 * HALF)
    ct = cj // HALF
    cp = cj % HALF
    for i in tl.range(pid, T // 2, npid):
        t0 = 2 * i
        pos0 = tl.load(P + t0 * PS)
        pos1 = tl.load(P + (t0 + 1) * PS)
        posv = tl.where(ct == 0, pos0, pos1)
        cos = tl.load(C + posv * CS0 + cp * CS1).to(tl.float32)
        sin = tl.load(C + posv * CS0 + (HALF + cp) * CS1).to(tl.float32)
        cosr = tl.reshape(cos, (2, 1, HALF))
        sinr = tl.reshape(sin, (2, 1, HALF))
        for hb in tl.static_range(QHB):
            _rope_merge_group(
                Q, QO, t0, hb * RQ, qw1, qw2, cosr, sinr,
                2, RQ, NQ, D, HALF, EPS, QS0,
            )
    ckj = tl.arange(0, KM * HALF)
    ckt = ckj // HALF
    ckp = ckj % HALF
    for i in tl.range(pid, T // KM, npid):
        t0 = KM * i
        if KM == 2:
            pos0 = tl.load(P + t0 * PS)
            pos1 = tl.load(P + (t0 + 1) * PS)
            posv = tl.where(ckt == 0, pos0, pos1)
        else:
            pos0 = tl.load(P + t0 * PS)
            pos1 = tl.load(P + (t0 + 1) * PS)
            pos2 = tl.load(P + (t0 + 2) * PS)
            pos3 = tl.load(P + (t0 + 3) * PS)
            posv = tl.where(
                ckt == 0, pos0,
                tl.where(ckt == 1, pos1, tl.where(ckt == 2, pos2, pos3)),
            )
        cos = tl.load(C + posv * CS0 + ckp * CS1).to(tl.float32)
        sin = tl.load(C + posv * CS0 + (HALF + ckp) * CS1).to(tl.float32)
        cosr = tl.reshape(cos, (KM, 1, HALF))
        sinr = tl.reshape(sin, (KM, 1, HALF))
        for hb in tl.static_range(KHB):
            _rope_merge_group(
                K, KO, t0, hb * RK, kw1, kw2, cosr, sinr,
                KM, RK, NK, D, HALF, EPS, KS0,
            )


def qk_gemma_rmsnorm_rope(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    cos_sin_cache,
    eps,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    t, qcols = q.shape
    nq = qcols // head_dim
    nk = k.shape[1] // head_dim
    qo = torch.empty_like(q, memory_format=torch.contiguous_format)
    ko = torch.empty_like(k, memory_format=torch.contiguous_format)
    nq_rows = t * nq
    nk_rows = t * nk
    total_rows = nq_rows + nk_rows
    half = rotary_dim // 2
    # Token-arm conditions: neox pairing, RoPE covers the full head dim,
    # power-of-two head dim / half, bf16, contiguous inputs, enough rows
    # to amortize, and head counts divisible by the group sizes (full
    # groups). fp16 falls back to the row-stream arm: the register
    # reshape path has a recorded silent-wrong-value precedent on fp16.
    use_token = (
        is_neox_style
        and rotary_dim == head_dim
        and head_dim > 1
        and (head_dim & (head_dim - 1)) == 0
        and half > 0
        and (half & (half - 1)) == 0
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and k.is_contiguous()
        and total_rows >= 128
    )
    if use_token:
        if t >= _TOKEN_FUSED_MIN_T:
            rq = min(32, nq & (-nq))
            rk = min(8, nk & (-nk))
        else:
            rq = min(8, nq & (-nq))
            rk = min(8, nk & (-nk))
        use_token = rq >= 4 and rk >= 4
    if use_token:
        if t >= _TOKEN_FUSED_MIN_T:
            # Merged arm only for the shape family it was measured on:
            # exactly one Q head-block and one K head-block.  Multi
            # head-block static_range inside the merged loops measured
            # catastrophic spills (nq=48 +373%, nq=64 +741%); those head
            # counts stay on the tokf arm at unchanged speed.
            if (
                t >= _TOKEN_MERGE_MIN_T
                and t % 2 == 0
                and rq == nq
                and rk == nk
            ):
                km = 4 if t % 4 == 0 else 2
                grid = (min(t // 2, _GRID_TARGET),)
                _fused_qk_gemma_tokm_kernel[grid](
                    q,
                    k,
                    q_weight,
                    k_weight,
                    cos_sin_cache,
                    positions,
                    qo,
                    ko,
                    t,
                    NQ=nq,
                    NK=nk,
                    D=head_dim,
                    EPS=eps,
                    QS0=q.stride(0),
                    KS0=k.stride(0),
                    QWS=q_weight.stride(0),
                    KWS=k_weight.stride(0),
                    CS0=cos_sin_cache.stride(0),
                    CS1=cos_sin_cache.stride(1),
                    PS=positions.stride(0),
                    RQ=rq,
                    RK=rk,
                    KM=km,
                    num_warps=_WARPS,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
                return (qo, ko)
            grid = (min(t, _GRID_TARGET),)
            _fused_qk_gemma_tokf_kernel[grid](
                q,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                qo,
                ko,
                t,
                NQ=nq,
                NK=nk,
                D=head_dim,
                EPS=eps,
                QS0=q.stride(0),
                KS0=k.stride(0),
                QWS=q_weight.stride(0),
                KWS=k_weight.stride(0),
                CS0=cos_sin_cache.stride(0),
                CS1=cos_sin_cache.stride(1),
                PS=positions.stride(0),
                RQ=rq,
                RK=rk,
                num_warps=_WARPS,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return (qo, ko)
        n_groups = t * (nq // rq) + t * (nk // rk)
        grid = (min(n_groups, _GRID_TARGET),)
        _fused_qk_gemma_token_kernel[grid](
            q,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            qo,
            ko,
            t,
            NQ=nq,
            NK=nk,
            D=head_dim,
            EPS=eps,
            QS0=q.stride(0),
            KS0=k.stride(0),
            QWS=q_weight.stride(0),
            KWS=k_weight.stride(0),
            CS0=cos_sin_cache.stride(0),
            CS1=cos_sin_cache.stride(1),
            PS=positions.stride(0),
            RQ=rq,
            RK=rk,
            num_warps=_WARPS,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return (qo, ko)
    rows = max(1, min(_ROWS, (total_rows + _GRID_TARGET - 1) // _GRID_TARGET))
    n_groups = triton.cdiv(nq_rows + nk_rows, rows)
    grid = (min(n_groups, _GRID_TARGET),)
    _fused_qk_gemma_rmsnorm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qo,
        ko,
        nq_rows,
        nk_rows,
        NQ=nq,
        NK=nk,
        D=head_dim,
        RD=rotary_dim,
        NEOX=is_neox_style,
        EPS=eps,
        QS0=q.stride(0),
        QS1=q.stride(1),
        KS0=k.stride(0),
        KS1=k.stride(1),
        QWS=q_weight.stride(0),
        KWS=k_weight.stride(0),
        CS0=cos_sin_cache.stride(0),
        CS1=cos_sin_cache.stride(1),
        PS=positions.stride(0),
        BP=triton.next_power_of_2(max(1, half)),
        BT=triton.next_power_of_2(max(1, head_dim - rotary_dim)),
        ROWS=rows,
        num_warps=_WARPS,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return (qo, ko)


__all__ = ["qk_gemma_rmsnorm_rope"]
