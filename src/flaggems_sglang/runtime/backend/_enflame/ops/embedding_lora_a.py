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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""embedding_lora_a (enflame GCU).

1-D affine rewrite of the original 2-D tile kernel: the 2-D gather/store
tile lowers to ``memref_ext.slice_deslice_start`` and fails fresh
compilation on the 2026-11-07 20:22 toolchain (see
compiler_issues/sglang_embedding_lora_a.md). All addresses here are
affine in the lane index (proven-safe form). MLP comes from a manually
interleaved longhand hot path: all token-id loads are issued before the
gathers, so the per-token (id load -> gather) dependency chains overlap
(the backend scheduler preserves source order and does not software
pipeline unrolled loops on its own).

The wrapper feeds token ids as a zero-copy little-endian low-word int32
view of the int64 input (probe-verified real 8-byte storage, stride 2;
fake-int64 or non-contiguous inputs fall back to the ``.to(int32)`` cast
kernel) — this removes the aten cast kernel from the device window while
keeping every path bit-correct.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _token_generic(
    rr,
    row,
    ids_ptr,
    w_ptr,
    extra_ptr,
    out_ptr,
    wi,
    active_rank,
    VOCAB: tl.constexpr,
    RANK: tl.constexpr,
    RANK_ACT: tl.constexpr,
    EXTRA_COUNT: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NEED_MASK: tl.constexpr,
    IDS_STRIDE: tl.constexpr,
):
    tok = tl.load(ids_ptr + row * IDS_STRIDE)
    if NEED_MASK:
        rrc = tl.minimum(rr, RANK_ACT - 1)
    else:
        rrc = rr
    if HAS_EXTRA:
        ctok = tl.minimum(tok, VOCAB - 1)
        v_main = tl.load(w_ptr + (wi * RANK_ACT + rrc) * VOCAB + ctok)
        etok = tl.maximum(tok - VOCAB, 0)
        v_extra = tl.load(
            extra_ptr + (wi * EXTRA_COUNT + etok) * RANK_ACT + rrc
        )
        v = tl.where(tok < VOCAB, v_main, v_extra)
    else:
        v = tl.load(w_ptr + (wi * RANK_ACT + rrc) * VOCAB + tok)
    if active_rank < RANK_ACT:
        v = tl.where(rr < active_rank, v, 0.0)
    if NEED_MASK:
        tl.store(out_ptr + row * RANK_ACT + rr, v, mask=rr < RANK_ACT)
    else:
        tl.store(out_ptr + row * RANK + rr, v)


@triton.jit
def _embedding_lora_a_kernel(
    ids_ptr,
    w_ptr,
    seg_indptr,
    widx_ptr,
    lranks_ptr,
    extra_ptr,
    out_ptr,
    token_blocks,
    total_programs,
    VOCAB: tl.constexpr,
    RANK: tl.constexpr,
    RANK_ACT: tl.constexpr,
    EXTRA_COUNT: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NEED_MASK: tl.constexpr,
    TPP: tl.constexpr,
    IDS_STRIDE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    for pid in range(pid0, total_programs, tl.num_programs(0)):
        segment = pid // token_blocks
        inner = pid - segment * token_blocks
        start = tl.load(seg_indptr + segment)
        end = tl.load(seg_indptr + segment + 1)
        wi = tl.load(widx_ptr + segment)
        active_rank = tl.load(lranks_ptr + wi)
        rr = tl.arange(0, RANK)
        tok0 = start + inner * TPP
        if NEED_MASK:
            for i in tl.static_range(TPP):
                row = tok0 + i
                if row < end:
                    _token_generic(
                        rr, row, ids_ptr, w_ptr, extra_ptr, out_ptr, wi,
                        active_rank, VOCAB, RANK, RANK_ACT, EXTRA_COUNT,
                        HAS_EXTRA, NEED_MASK, IDS_STRIDE,
                    )
        elif tok0 + TPP <= end:
            wbase = (wi * RANK_ACT + rr) * VOCAB
            xbase = wi * EXTRA_COUNT * RANK_ACT
            tbase = tok0 * IDS_STRIDE
            # token id loads (issued together; independent)
            t0 = tl.load(ids_ptr + tbase + 0 * IDS_STRIDE)
            t1 = tl.load(ids_ptr + tbase + 1 * IDS_STRIDE)
            t2 = tl.load(ids_ptr + tbase + 2 * IDS_STRIDE)
            t3 = tl.load(ids_ptr + tbase + 3 * IDS_STRIDE)
            t4 = tl.load(ids_ptr + tbase + 4 * IDS_STRIDE)
            t5 = tl.load(ids_ptr + tbase + 5 * IDS_STRIDE)
            t6 = tl.load(ids_ptr + tbase + 6 * IDS_STRIDE)
            t7 = tl.load(ids_ptr + tbase + 7 * IDS_STRIDE)
            t8 = tl.load(ids_ptr + tbase + 8 * IDS_STRIDE)
            t9 = tl.load(ids_ptr + tbase + 9 * IDS_STRIDE)
            t10 = tl.load(ids_ptr + tbase + 10 * IDS_STRIDE)
            t11 = tl.load(ids_ptr + tbase + 11 * IDS_STRIDE)
            t12 = tl.load(ids_ptr + tbase + 12 * IDS_STRIDE)
            t13 = tl.load(ids_ptr + tbase + 13 * IDS_STRIDE)
            t14 = tl.load(ids_ptr + tbase + 14 * IDS_STRIDE)
            t15 = tl.load(ids_ptr + tbase + 15 * IDS_STRIDE)
            t16 = tl.load(ids_ptr + tbase + 16 * IDS_STRIDE)
            t17 = tl.load(ids_ptr + tbase + 17 * IDS_STRIDE)
            t18 = tl.load(ids_ptr + tbase + 18 * IDS_STRIDE)
            t19 = tl.load(ids_ptr + tbase + 19 * IDS_STRIDE)
            t20 = tl.load(ids_ptr + tbase + 20 * IDS_STRIDE)
            t21 = tl.load(ids_ptr + tbase + 21 * IDS_STRIDE)
            t22 = tl.load(ids_ptr + tbase + 22 * IDS_STRIDE)
            t23 = tl.load(ids_ptr + tbase + 23 * IDS_STRIDE)
            t24 = tl.load(ids_ptr + tbase + 24 * IDS_STRIDE)
            t25 = tl.load(ids_ptr + tbase + 25 * IDS_STRIDE)
            t26 = tl.load(ids_ptr + tbase + 26 * IDS_STRIDE)
            t27 = tl.load(ids_ptr + tbase + 27 * IDS_STRIDE)
            t28 = tl.load(ids_ptr + tbase + 28 * IDS_STRIDE)
            t29 = tl.load(ids_ptr + tbase + 29 * IDS_STRIDE)
            t30 = tl.load(ids_ptr + tbase + 30 * IDS_STRIDE)
            t31 = tl.load(ids_ptr + tbase + 31 * IDS_STRIDE)
            t32 = tl.load(ids_ptr + tbase + 32 * IDS_STRIDE)
            t33 = tl.load(ids_ptr + tbase + 33 * IDS_STRIDE)
            t34 = tl.load(ids_ptr + tbase + 34 * IDS_STRIDE)
            t35 = tl.load(ids_ptr + tbase + 35 * IDS_STRIDE)
            t36 = tl.load(ids_ptr + tbase + 36 * IDS_STRIDE)
            t37 = tl.load(ids_ptr + tbase + 37 * IDS_STRIDE)
            t38 = tl.load(ids_ptr + tbase + 38 * IDS_STRIDE)
            t39 = tl.load(ids_ptr + tbase + 39 * IDS_STRIDE)
            t40 = tl.load(ids_ptr + tbase + 40 * IDS_STRIDE)
            t41 = tl.load(ids_ptr + tbase + 41 * IDS_STRIDE)
            t42 = tl.load(ids_ptr + tbase + 42 * IDS_STRIDE)
            t43 = tl.load(ids_ptr + tbase + 43 * IDS_STRIDE)
            t44 = tl.load(ids_ptr + tbase + 44 * IDS_STRIDE)
            t45 = tl.load(ids_ptr + tbase + 45 * IDS_STRIDE)
            t46 = tl.load(ids_ptr + tbase + 46 * IDS_STRIDE)
            t47 = tl.load(ids_ptr + tbase + 47 * IDS_STRIDE)
            t48 = tl.load(ids_ptr + tbase + 48 * IDS_STRIDE)
            t49 = tl.load(ids_ptr + tbase + 49 * IDS_STRIDE)
            t50 = tl.load(ids_ptr + tbase + 50 * IDS_STRIDE)
            t51 = tl.load(ids_ptr + tbase + 51 * IDS_STRIDE)
            t52 = tl.load(ids_ptr + tbase + 52 * IDS_STRIDE)
            t53 = tl.load(ids_ptr + tbase + 53 * IDS_STRIDE)
            t54 = tl.load(ids_ptr + tbase + 54 * IDS_STRIDE)
            t55 = tl.load(ids_ptr + tbase + 55 * IDS_STRIDE)
            t56 = tl.load(ids_ptr + tbase + 56 * IDS_STRIDE)
            t57 = tl.load(ids_ptr + tbase + 57 * IDS_STRIDE)
            t58 = tl.load(ids_ptr + tbase + 58 * IDS_STRIDE)
            t59 = tl.load(ids_ptr + tbase + 59 * IDS_STRIDE)
            t60 = tl.load(ids_ptr + tbase + 60 * IDS_STRIDE)
            t61 = tl.load(ids_ptr + tbase + 61 * IDS_STRIDE)
            t62 = tl.load(ids_ptr + tbase + 62 * IDS_STRIDE)
            t63 = tl.load(ids_ptr + tbase + 63 * IDS_STRIDE)
            # gathers (each depends only on its own id load)
            if HAS_EXTRA:
                ct0 = tl.minimum(t0, VOCAB - 1)
                vm0 = tl.load(w_ptr + wbase + ct0)
                et0 = tl.maximum(t0 - VOCAB, 0)
                ve0 = tl.load(extra_ptr + xbase + et0 * RANK_ACT + rr)
                v0 = tl.where(t0 < VOCAB, vm0, ve0)
                ct1 = tl.minimum(t1, VOCAB - 1)
                vm1 = tl.load(w_ptr + wbase + ct1)
                et1 = tl.maximum(t1 - VOCAB, 0)
                ve1 = tl.load(extra_ptr + xbase + et1 * RANK_ACT + rr)
                v1 = tl.where(t1 < VOCAB, vm1, ve1)
                ct2 = tl.minimum(t2, VOCAB - 1)
                vm2 = tl.load(w_ptr + wbase + ct2)
                et2 = tl.maximum(t2 - VOCAB, 0)
                ve2 = tl.load(extra_ptr + xbase + et2 * RANK_ACT + rr)
                v2 = tl.where(t2 < VOCAB, vm2, ve2)
                ct3 = tl.minimum(t3, VOCAB - 1)
                vm3 = tl.load(w_ptr + wbase + ct3)
                et3 = tl.maximum(t3 - VOCAB, 0)
                ve3 = tl.load(extra_ptr + xbase + et3 * RANK_ACT + rr)
                v3 = tl.where(t3 < VOCAB, vm3, ve3)
                ct4 = tl.minimum(t4, VOCAB - 1)
                vm4 = tl.load(w_ptr + wbase + ct4)
                et4 = tl.maximum(t4 - VOCAB, 0)
                ve4 = tl.load(extra_ptr + xbase + et4 * RANK_ACT + rr)
                v4 = tl.where(t4 < VOCAB, vm4, ve4)
                ct5 = tl.minimum(t5, VOCAB - 1)
                vm5 = tl.load(w_ptr + wbase + ct5)
                et5 = tl.maximum(t5 - VOCAB, 0)
                ve5 = tl.load(extra_ptr + xbase + et5 * RANK_ACT + rr)
                v5 = tl.where(t5 < VOCAB, vm5, ve5)
                ct6 = tl.minimum(t6, VOCAB - 1)
                vm6 = tl.load(w_ptr + wbase + ct6)
                et6 = tl.maximum(t6 - VOCAB, 0)
                ve6 = tl.load(extra_ptr + xbase + et6 * RANK_ACT + rr)
                v6 = tl.where(t6 < VOCAB, vm6, ve6)
                ct7 = tl.minimum(t7, VOCAB - 1)
                vm7 = tl.load(w_ptr + wbase + ct7)
                et7 = tl.maximum(t7 - VOCAB, 0)
                ve7 = tl.load(extra_ptr + xbase + et7 * RANK_ACT + rr)
                v7 = tl.where(t7 < VOCAB, vm7, ve7)
                ct8 = tl.minimum(t8, VOCAB - 1)
                vm8 = tl.load(w_ptr + wbase + ct8)
                et8 = tl.maximum(t8 - VOCAB, 0)
                ve8 = tl.load(extra_ptr + xbase + et8 * RANK_ACT + rr)
                v8 = tl.where(t8 < VOCAB, vm8, ve8)
                ct9 = tl.minimum(t9, VOCAB - 1)
                vm9 = tl.load(w_ptr + wbase + ct9)
                et9 = tl.maximum(t9 - VOCAB, 0)
                ve9 = tl.load(extra_ptr + xbase + et9 * RANK_ACT + rr)
                v9 = tl.where(t9 < VOCAB, vm9, ve9)
                ct10 = tl.minimum(t10, VOCAB - 1)
                vm10 = tl.load(w_ptr + wbase + ct10)
                et10 = tl.maximum(t10 - VOCAB, 0)
                ve10 = tl.load(extra_ptr + xbase + et10 * RANK_ACT + rr)
                v10 = tl.where(t10 < VOCAB, vm10, ve10)
                ct11 = tl.minimum(t11, VOCAB - 1)
                vm11 = tl.load(w_ptr + wbase + ct11)
                et11 = tl.maximum(t11 - VOCAB, 0)
                ve11 = tl.load(extra_ptr + xbase + et11 * RANK_ACT + rr)
                v11 = tl.where(t11 < VOCAB, vm11, ve11)
                ct12 = tl.minimum(t12, VOCAB - 1)
                vm12 = tl.load(w_ptr + wbase + ct12)
                et12 = tl.maximum(t12 - VOCAB, 0)
                ve12 = tl.load(extra_ptr + xbase + et12 * RANK_ACT + rr)
                v12 = tl.where(t12 < VOCAB, vm12, ve12)
                ct13 = tl.minimum(t13, VOCAB - 1)
                vm13 = tl.load(w_ptr + wbase + ct13)
                et13 = tl.maximum(t13 - VOCAB, 0)
                ve13 = tl.load(extra_ptr + xbase + et13 * RANK_ACT + rr)
                v13 = tl.where(t13 < VOCAB, vm13, ve13)
                ct14 = tl.minimum(t14, VOCAB - 1)
                vm14 = tl.load(w_ptr + wbase + ct14)
                et14 = tl.maximum(t14 - VOCAB, 0)
                ve14 = tl.load(extra_ptr + xbase + et14 * RANK_ACT + rr)
                v14 = tl.where(t14 < VOCAB, vm14, ve14)
                ct15 = tl.minimum(t15, VOCAB - 1)
                vm15 = tl.load(w_ptr + wbase + ct15)
                et15 = tl.maximum(t15 - VOCAB, 0)
                ve15 = tl.load(extra_ptr + xbase + et15 * RANK_ACT + rr)
                v15 = tl.where(t15 < VOCAB, vm15, ve15)
                ct16 = tl.minimum(t16, VOCAB - 1)
                vm16 = tl.load(w_ptr + wbase + ct16)
                et16 = tl.maximum(t16 - VOCAB, 0)
                ve16 = tl.load(extra_ptr + xbase + et16 * RANK_ACT + rr)
                v16 = tl.where(t16 < VOCAB, vm16, ve16)
                ct17 = tl.minimum(t17, VOCAB - 1)
                vm17 = tl.load(w_ptr + wbase + ct17)
                et17 = tl.maximum(t17 - VOCAB, 0)
                ve17 = tl.load(extra_ptr + xbase + et17 * RANK_ACT + rr)
                v17 = tl.where(t17 < VOCAB, vm17, ve17)
                ct18 = tl.minimum(t18, VOCAB - 1)
                vm18 = tl.load(w_ptr + wbase + ct18)
                et18 = tl.maximum(t18 - VOCAB, 0)
                ve18 = tl.load(extra_ptr + xbase + et18 * RANK_ACT + rr)
                v18 = tl.where(t18 < VOCAB, vm18, ve18)
                ct19 = tl.minimum(t19, VOCAB - 1)
                vm19 = tl.load(w_ptr + wbase + ct19)
                et19 = tl.maximum(t19 - VOCAB, 0)
                ve19 = tl.load(extra_ptr + xbase + et19 * RANK_ACT + rr)
                v19 = tl.where(t19 < VOCAB, vm19, ve19)
                ct20 = tl.minimum(t20, VOCAB - 1)
                vm20 = tl.load(w_ptr + wbase + ct20)
                et20 = tl.maximum(t20 - VOCAB, 0)
                ve20 = tl.load(extra_ptr + xbase + et20 * RANK_ACT + rr)
                v20 = tl.where(t20 < VOCAB, vm20, ve20)
                ct21 = tl.minimum(t21, VOCAB - 1)
                vm21 = tl.load(w_ptr + wbase + ct21)
                et21 = tl.maximum(t21 - VOCAB, 0)
                ve21 = tl.load(extra_ptr + xbase + et21 * RANK_ACT + rr)
                v21 = tl.where(t21 < VOCAB, vm21, ve21)
                ct22 = tl.minimum(t22, VOCAB - 1)
                vm22 = tl.load(w_ptr + wbase + ct22)
                et22 = tl.maximum(t22 - VOCAB, 0)
                ve22 = tl.load(extra_ptr + xbase + et22 * RANK_ACT + rr)
                v22 = tl.where(t22 < VOCAB, vm22, ve22)
                ct23 = tl.minimum(t23, VOCAB - 1)
                vm23 = tl.load(w_ptr + wbase + ct23)
                et23 = tl.maximum(t23 - VOCAB, 0)
                ve23 = tl.load(extra_ptr + xbase + et23 * RANK_ACT + rr)
                v23 = tl.where(t23 < VOCAB, vm23, ve23)
                ct24 = tl.minimum(t24, VOCAB - 1)
                vm24 = tl.load(w_ptr + wbase + ct24)
                et24 = tl.maximum(t24 - VOCAB, 0)
                ve24 = tl.load(extra_ptr + xbase + et24 * RANK_ACT + rr)
                v24 = tl.where(t24 < VOCAB, vm24, ve24)
                ct25 = tl.minimum(t25, VOCAB - 1)
                vm25 = tl.load(w_ptr + wbase + ct25)
                et25 = tl.maximum(t25 - VOCAB, 0)
                ve25 = tl.load(extra_ptr + xbase + et25 * RANK_ACT + rr)
                v25 = tl.where(t25 < VOCAB, vm25, ve25)
                ct26 = tl.minimum(t26, VOCAB - 1)
                vm26 = tl.load(w_ptr + wbase + ct26)
                et26 = tl.maximum(t26 - VOCAB, 0)
                ve26 = tl.load(extra_ptr + xbase + et26 * RANK_ACT + rr)
                v26 = tl.where(t26 < VOCAB, vm26, ve26)
                ct27 = tl.minimum(t27, VOCAB - 1)
                vm27 = tl.load(w_ptr + wbase + ct27)
                et27 = tl.maximum(t27 - VOCAB, 0)
                ve27 = tl.load(extra_ptr + xbase + et27 * RANK_ACT + rr)
                v27 = tl.where(t27 < VOCAB, vm27, ve27)
                ct28 = tl.minimum(t28, VOCAB - 1)
                vm28 = tl.load(w_ptr + wbase + ct28)
                et28 = tl.maximum(t28 - VOCAB, 0)
                ve28 = tl.load(extra_ptr + xbase + et28 * RANK_ACT + rr)
                v28 = tl.where(t28 < VOCAB, vm28, ve28)
                ct29 = tl.minimum(t29, VOCAB - 1)
                vm29 = tl.load(w_ptr + wbase + ct29)
                et29 = tl.maximum(t29 - VOCAB, 0)
                ve29 = tl.load(extra_ptr + xbase + et29 * RANK_ACT + rr)
                v29 = tl.where(t29 < VOCAB, vm29, ve29)
                ct30 = tl.minimum(t30, VOCAB - 1)
                vm30 = tl.load(w_ptr + wbase + ct30)
                et30 = tl.maximum(t30 - VOCAB, 0)
                ve30 = tl.load(extra_ptr + xbase + et30 * RANK_ACT + rr)
                v30 = tl.where(t30 < VOCAB, vm30, ve30)
                ct31 = tl.minimum(t31, VOCAB - 1)
                vm31 = tl.load(w_ptr + wbase + ct31)
                et31 = tl.maximum(t31 - VOCAB, 0)
                ve31 = tl.load(extra_ptr + xbase + et31 * RANK_ACT + rr)
                v31 = tl.where(t31 < VOCAB, vm31, ve31)
                ct32 = tl.minimum(t32, VOCAB - 1)
                vm32 = tl.load(w_ptr + wbase + ct32)
                et32 = tl.maximum(t32 - VOCAB, 0)
                ve32 = tl.load(extra_ptr + xbase + et32 * RANK_ACT + rr)
                v32 = tl.where(t32 < VOCAB, vm32, ve32)
                ct33 = tl.minimum(t33, VOCAB - 1)
                vm33 = tl.load(w_ptr + wbase + ct33)
                et33 = tl.maximum(t33 - VOCAB, 0)
                ve33 = tl.load(extra_ptr + xbase + et33 * RANK_ACT + rr)
                v33 = tl.where(t33 < VOCAB, vm33, ve33)
                ct34 = tl.minimum(t34, VOCAB - 1)
                vm34 = tl.load(w_ptr + wbase + ct34)
                et34 = tl.maximum(t34 - VOCAB, 0)
                ve34 = tl.load(extra_ptr + xbase + et34 * RANK_ACT + rr)
                v34 = tl.where(t34 < VOCAB, vm34, ve34)
                ct35 = tl.minimum(t35, VOCAB - 1)
                vm35 = tl.load(w_ptr + wbase + ct35)
                et35 = tl.maximum(t35 - VOCAB, 0)
                ve35 = tl.load(extra_ptr + xbase + et35 * RANK_ACT + rr)
                v35 = tl.where(t35 < VOCAB, vm35, ve35)
                ct36 = tl.minimum(t36, VOCAB - 1)
                vm36 = tl.load(w_ptr + wbase + ct36)
                et36 = tl.maximum(t36 - VOCAB, 0)
                ve36 = tl.load(extra_ptr + xbase + et36 * RANK_ACT + rr)
                v36 = tl.where(t36 < VOCAB, vm36, ve36)
                ct37 = tl.minimum(t37, VOCAB - 1)
                vm37 = tl.load(w_ptr + wbase + ct37)
                et37 = tl.maximum(t37 - VOCAB, 0)
                ve37 = tl.load(extra_ptr + xbase + et37 * RANK_ACT + rr)
                v37 = tl.where(t37 < VOCAB, vm37, ve37)
                ct38 = tl.minimum(t38, VOCAB - 1)
                vm38 = tl.load(w_ptr + wbase + ct38)
                et38 = tl.maximum(t38 - VOCAB, 0)
                ve38 = tl.load(extra_ptr + xbase + et38 * RANK_ACT + rr)
                v38 = tl.where(t38 < VOCAB, vm38, ve38)
                ct39 = tl.minimum(t39, VOCAB - 1)
                vm39 = tl.load(w_ptr + wbase + ct39)
                et39 = tl.maximum(t39 - VOCAB, 0)
                ve39 = tl.load(extra_ptr + xbase + et39 * RANK_ACT + rr)
                v39 = tl.where(t39 < VOCAB, vm39, ve39)
                ct40 = tl.minimum(t40, VOCAB - 1)
                vm40 = tl.load(w_ptr + wbase + ct40)
                et40 = tl.maximum(t40 - VOCAB, 0)
                ve40 = tl.load(extra_ptr + xbase + et40 * RANK_ACT + rr)
                v40 = tl.where(t40 < VOCAB, vm40, ve40)
                ct41 = tl.minimum(t41, VOCAB - 1)
                vm41 = tl.load(w_ptr + wbase + ct41)
                et41 = tl.maximum(t41 - VOCAB, 0)
                ve41 = tl.load(extra_ptr + xbase + et41 * RANK_ACT + rr)
                v41 = tl.where(t41 < VOCAB, vm41, ve41)
                ct42 = tl.minimum(t42, VOCAB - 1)
                vm42 = tl.load(w_ptr + wbase + ct42)
                et42 = tl.maximum(t42 - VOCAB, 0)
                ve42 = tl.load(extra_ptr + xbase + et42 * RANK_ACT + rr)
                v42 = tl.where(t42 < VOCAB, vm42, ve42)
                ct43 = tl.minimum(t43, VOCAB - 1)
                vm43 = tl.load(w_ptr + wbase + ct43)
                et43 = tl.maximum(t43 - VOCAB, 0)
                ve43 = tl.load(extra_ptr + xbase + et43 * RANK_ACT + rr)
                v43 = tl.where(t43 < VOCAB, vm43, ve43)
                ct44 = tl.minimum(t44, VOCAB - 1)
                vm44 = tl.load(w_ptr + wbase + ct44)
                et44 = tl.maximum(t44 - VOCAB, 0)
                ve44 = tl.load(extra_ptr + xbase + et44 * RANK_ACT + rr)
                v44 = tl.where(t44 < VOCAB, vm44, ve44)
                ct45 = tl.minimum(t45, VOCAB - 1)
                vm45 = tl.load(w_ptr + wbase + ct45)
                et45 = tl.maximum(t45 - VOCAB, 0)
                ve45 = tl.load(extra_ptr + xbase + et45 * RANK_ACT + rr)
                v45 = tl.where(t45 < VOCAB, vm45, ve45)
                ct46 = tl.minimum(t46, VOCAB - 1)
                vm46 = tl.load(w_ptr + wbase + ct46)
                et46 = tl.maximum(t46 - VOCAB, 0)
                ve46 = tl.load(extra_ptr + xbase + et46 * RANK_ACT + rr)
                v46 = tl.where(t46 < VOCAB, vm46, ve46)
                ct47 = tl.minimum(t47, VOCAB - 1)
                vm47 = tl.load(w_ptr + wbase + ct47)
                et47 = tl.maximum(t47 - VOCAB, 0)
                ve47 = tl.load(extra_ptr + xbase + et47 * RANK_ACT + rr)
                v47 = tl.where(t47 < VOCAB, vm47, ve47)
                ct48 = tl.minimum(t48, VOCAB - 1)
                vm48 = tl.load(w_ptr + wbase + ct48)
                et48 = tl.maximum(t48 - VOCAB, 0)
                ve48 = tl.load(extra_ptr + xbase + et48 * RANK_ACT + rr)
                v48 = tl.where(t48 < VOCAB, vm48, ve48)
                ct49 = tl.minimum(t49, VOCAB - 1)
                vm49 = tl.load(w_ptr + wbase + ct49)
                et49 = tl.maximum(t49 - VOCAB, 0)
                ve49 = tl.load(extra_ptr + xbase + et49 * RANK_ACT + rr)
                v49 = tl.where(t49 < VOCAB, vm49, ve49)
                ct50 = tl.minimum(t50, VOCAB - 1)
                vm50 = tl.load(w_ptr + wbase + ct50)
                et50 = tl.maximum(t50 - VOCAB, 0)
                ve50 = tl.load(extra_ptr + xbase + et50 * RANK_ACT + rr)
                v50 = tl.where(t50 < VOCAB, vm50, ve50)
                ct51 = tl.minimum(t51, VOCAB - 1)
                vm51 = tl.load(w_ptr + wbase + ct51)
                et51 = tl.maximum(t51 - VOCAB, 0)
                ve51 = tl.load(extra_ptr + xbase + et51 * RANK_ACT + rr)
                v51 = tl.where(t51 < VOCAB, vm51, ve51)
                ct52 = tl.minimum(t52, VOCAB - 1)
                vm52 = tl.load(w_ptr + wbase + ct52)
                et52 = tl.maximum(t52 - VOCAB, 0)
                ve52 = tl.load(extra_ptr + xbase + et52 * RANK_ACT + rr)
                v52 = tl.where(t52 < VOCAB, vm52, ve52)
                ct53 = tl.minimum(t53, VOCAB - 1)
                vm53 = tl.load(w_ptr + wbase + ct53)
                et53 = tl.maximum(t53 - VOCAB, 0)
                ve53 = tl.load(extra_ptr + xbase + et53 * RANK_ACT + rr)
                v53 = tl.where(t53 < VOCAB, vm53, ve53)
                ct54 = tl.minimum(t54, VOCAB - 1)
                vm54 = tl.load(w_ptr + wbase + ct54)
                et54 = tl.maximum(t54 - VOCAB, 0)
                ve54 = tl.load(extra_ptr + xbase + et54 * RANK_ACT + rr)
                v54 = tl.where(t54 < VOCAB, vm54, ve54)
                ct55 = tl.minimum(t55, VOCAB - 1)
                vm55 = tl.load(w_ptr + wbase + ct55)
                et55 = tl.maximum(t55 - VOCAB, 0)
                ve55 = tl.load(extra_ptr + xbase + et55 * RANK_ACT + rr)
                v55 = tl.where(t55 < VOCAB, vm55, ve55)
                ct56 = tl.minimum(t56, VOCAB - 1)
                vm56 = tl.load(w_ptr + wbase + ct56)
                et56 = tl.maximum(t56 - VOCAB, 0)
                ve56 = tl.load(extra_ptr + xbase + et56 * RANK_ACT + rr)
                v56 = tl.where(t56 < VOCAB, vm56, ve56)
                ct57 = tl.minimum(t57, VOCAB - 1)
                vm57 = tl.load(w_ptr + wbase + ct57)
                et57 = tl.maximum(t57 - VOCAB, 0)
                ve57 = tl.load(extra_ptr + xbase + et57 * RANK_ACT + rr)
                v57 = tl.where(t57 < VOCAB, vm57, ve57)
                ct58 = tl.minimum(t58, VOCAB - 1)
                vm58 = tl.load(w_ptr + wbase + ct58)
                et58 = tl.maximum(t58 - VOCAB, 0)
                ve58 = tl.load(extra_ptr + xbase + et58 * RANK_ACT + rr)
                v58 = tl.where(t58 < VOCAB, vm58, ve58)
                ct59 = tl.minimum(t59, VOCAB - 1)
                vm59 = tl.load(w_ptr + wbase + ct59)
                et59 = tl.maximum(t59 - VOCAB, 0)
                ve59 = tl.load(extra_ptr + xbase + et59 * RANK_ACT + rr)
                v59 = tl.where(t59 < VOCAB, vm59, ve59)
                ct60 = tl.minimum(t60, VOCAB - 1)
                vm60 = tl.load(w_ptr + wbase + ct60)
                et60 = tl.maximum(t60 - VOCAB, 0)
                ve60 = tl.load(extra_ptr + xbase + et60 * RANK_ACT + rr)
                v60 = tl.where(t60 < VOCAB, vm60, ve60)
                ct61 = tl.minimum(t61, VOCAB - 1)
                vm61 = tl.load(w_ptr + wbase + ct61)
                et61 = tl.maximum(t61 - VOCAB, 0)
                ve61 = tl.load(extra_ptr + xbase + et61 * RANK_ACT + rr)
                v61 = tl.where(t61 < VOCAB, vm61, ve61)
                ct62 = tl.minimum(t62, VOCAB - 1)
                vm62 = tl.load(w_ptr + wbase + ct62)
                et62 = tl.maximum(t62 - VOCAB, 0)
                ve62 = tl.load(extra_ptr + xbase + et62 * RANK_ACT + rr)
                v62 = tl.where(t62 < VOCAB, vm62, ve62)
                ct63 = tl.minimum(t63, VOCAB - 1)
                vm63 = tl.load(w_ptr + wbase + ct63)
                et63 = tl.maximum(t63 - VOCAB, 0)
                ve63 = tl.load(extra_ptr + xbase + et63 * RANK_ACT + rr)
                v63 = tl.where(t63 < VOCAB, vm63, ve63)
            else:
                v0 = tl.load(w_ptr + wbase + t0)
                v1 = tl.load(w_ptr + wbase + t1)
                v2 = tl.load(w_ptr + wbase + t2)
                v3 = tl.load(w_ptr + wbase + t3)
                v4 = tl.load(w_ptr + wbase + t4)
                v5 = tl.load(w_ptr + wbase + t5)
                v6 = tl.load(w_ptr + wbase + t6)
                v7 = tl.load(w_ptr + wbase + t7)
                v8 = tl.load(w_ptr + wbase + t8)
                v9 = tl.load(w_ptr + wbase + t9)
                v10 = tl.load(w_ptr + wbase + t10)
                v11 = tl.load(w_ptr + wbase + t11)
                v12 = tl.load(w_ptr + wbase + t12)
                v13 = tl.load(w_ptr + wbase + t13)
                v14 = tl.load(w_ptr + wbase + t14)
                v15 = tl.load(w_ptr + wbase + t15)
                v16 = tl.load(w_ptr + wbase + t16)
                v17 = tl.load(w_ptr + wbase + t17)
                v18 = tl.load(w_ptr + wbase + t18)
                v19 = tl.load(w_ptr + wbase + t19)
                v20 = tl.load(w_ptr + wbase + t20)
                v21 = tl.load(w_ptr + wbase + t21)
                v22 = tl.load(w_ptr + wbase + t22)
                v23 = tl.load(w_ptr + wbase + t23)
                v24 = tl.load(w_ptr + wbase + t24)
                v25 = tl.load(w_ptr + wbase + t25)
                v26 = tl.load(w_ptr + wbase + t26)
                v27 = tl.load(w_ptr + wbase + t27)
                v28 = tl.load(w_ptr + wbase + t28)
                v29 = tl.load(w_ptr + wbase + t29)
                v30 = tl.load(w_ptr + wbase + t30)
                v31 = tl.load(w_ptr + wbase + t31)
                v32 = tl.load(w_ptr + wbase + t32)
                v33 = tl.load(w_ptr + wbase + t33)
                v34 = tl.load(w_ptr + wbase + t34)
                v35 = tl.load(w_ptr + wbase + t35)
                v36 = tl.load(w_ptr + wbase + t36)
                v37 = tl.load(w_ptr + wbase + t37)
                v38 = tl.load(w_ptr + wbase + t38)
                v39 = tl.load(w_ptr + wbase + t39)
                v40 = tl.load(w_ptr + wbase + t40)
                v41 = tl.load(w_ptr + wbase + t41)
                v42 = tl.load(w_ptr + wbase + t42)
                v43 = tl.load(w_ptr + wbase + t43)
                v44 = tl.load(w_ptr + wbase + t44)
                v45 = tl.load(w_ptr + wbase + t45)
                v46 = tl.load(w_ptr + wbase + t46)
                v47 = tl.load(w_ptr + wbase + t47)
                v48 = tl.load(w_ptr + wbase + t48)
                v49 = tl.load(w_ptr + wbase + t49)
                v50 = tl.load(w_ptr + wbase + t50)
                v51 = tl.load(w_ptr + wbase + t51)
                v52 = tl.load(w_ptr + wbase + t52)
                v53 = tl.load(w_ptr + wbase + t53)
                v54 = tl.load(w_ptr + wbase + t54)
                v55 = tl.load(w_ptr + wbase + t55)
                v56 = tl.load(w_ptr + wbase + t56)
                v57 = tl.load(w_ptr + wbase + t57)
                v58 = tl.load(w_ptr + wbase + t58)
                v59 = tl.load(w_ptr + wbase + t59)
                v60 = tl.load(w_ptr + wbase + t60)
                v61 = tl.load(w_ptr + wbase + t61)
                v62 = tl.load(w_ptr + wbase + t62)
                v63 = tl.load(w_ptr + wbase + t63)
            # zero ranks >= active_rank (per-segment lora rank)
            if active_rank < RANK_ACT:
                zmask = rr >= active_rank
                v0 = tl.where(zmask, 0.0, v0)
                v1 = tl.where(zmask, 0.0, v1)
                v2 = tl.where(zmask, 0.0, v2)
                v3 = tl.where(zmask, 0.0, v3)
                v4 = tl.where(zmask, 0.0, v4)
                v5 = tl.where(zmask, 0.0, v5)
                v6 = tl.where(zmask, 0.0, v6)
                v7 = tl.where(zmask, 0.0, v7)
                v8 = tl.where(zmask, 0.0, v8)
                v9 = tl.where(zmask, 0.0, v9)
                v10 = tl.where(zmask, 0.0, v10)
                v11 = tl.where(zmask, 0.0, v11)
                v12 = tl.where(zmask, 0.0, v12)
                v13 = tl.where(zmask, 0.0, v13)
                v14 = tl.where(zmask, 0.0, v14)
                v15 = tl.where(zmask, 0.0, v15)
                v16 = tl.where(zmask, 0.0, v16)
                v17 = tl.where(zmask, 0.0, v17)
                v18 = tl.where(zmask, 0.0, v18)
                v19 = tl.where(zmask, 0.0, v19)
                v20 = tl.where(zmask, 0.0, v20)
                v21 = tl.where(zmask, 0.0, v21)
                v22 = tl.where(zmask, 0.0, v22)
                v23 = tl.where(zmask, 0.0, v23)
                v24 = tl.where(zmask, 0.0, v24)
                v25 = tl.where(zmask, 0.0, v25)
                v26 = tl.where(zmask, 0.0, v26)
                v27 = tl.where(zmask, 0.0, v27)
                v28 = tl.where(zmask, 0.0, v28)
                v29 = tl.where(zmask, 0.0, v29)
                v30 = tl.where(zmask, 0.0, v30)
                v31 = tl.where(zmask, 0.0, v31)
                v32 = tl.where(zmask, 0.0, v32)
                v33 = tl.where(zmask, 0.0, v33)
                v34 = tl.where(zmask, 0.0, v34)
                v35 = tl.where(zmask, 0.0, v35)
                v36 = tl.where(zmask, 0.0, v36)
                v37 = tl.where(zmask, 0.0, v37)
                v38 = tl.where(zmask, 0.0, v38)
                v39 = tl.where(zmask, 0.0, v39)
                v40 = tl.where(zmask, 0.0, v40)
                v41 = tl.where(zmask, 0.0, v41)
                v42 = tl.where(zmask, 0.0, v42)
                v43 = tl.where(zmask, 0.0, v43)
                v44 = tl.where(zmask, 0.0, v44)
                v45 = tl.where(zmask, 0.0, v45)
                v46 = tl.where(zmask, 0.0, v46)
                v47 = tl.where(zmask, 0.0, v47)
                v48 = tl.where(zmask, 0.0, v48)
                v49 = tl.where(zmask, 0.0, v49)
                v50 = tl.where(zmask, 0.0, v50)
                v51 = tl.where(zmask, 0.0, v51)
                v52 = tl.where(zmask, 0.0, v52)
                v53 = tl.where(zmask, 0.0, v53)
                v54 = tl.where(zmask, 0.0, v54)
                v55 = tl.where(zmask, 0.0, v55)
                v56 = tl.where(zmask, 0.0, v56)
                v57 = tl.where(zmask, 0.0, v57)
                v58 = tl.where(zmask, 0.0, v58)
                v59 = tl.where(zmask, 0.0, v59)
                v60 = tl.where(zmask, 0.0, v60)
                v61 = tl.where(zmask, 0.0, v61)
                v62 = tl.where(zmask, 0.0, v62)
                v63 = tl.where(zmask, 0.0, v63)
            # contiguous per-row stores
            tl.store(out_ptr + (tok0 + 0) * RANK + rr, v0)
            tl.store(out_ptr + (tok0 + 1) * RANK + rr, v1)
            tl.store(out_ptr + (tok0 + 2) * RANK + rr, v2)
            tl.store(out_ptr + (tok0 + 3) * RANK + rr, v3)
            tl.store(out_ptr + (tok0 + 4) * RANK + rr, v4)
            tl.store(out_ptr + (tok0 + 5) * RANK + rr, v5)
            tl.store(out_ptr + (tok0 + 6) * RANK + rr, v6)
            tl.store(out_ptr + (tok0 + 7) * RANK + rr, v7)
            tl.store(out_ptr + (tok0 + 8) * RANK + rr, v8)
            tl.store(out_ptr + (tok0 + 9) * RANK + rr, v9)
            tl.store(out_ptr + (tok0 + 10) * RANK + rr, v10)
            tl.store(out_ptr + (tok0 + 11) * RANK + rr, v11)
            tl.store(out_ptr + (tok0 + 12) * RANK + rr, v12)
            tl.store(out_ptr + (tok0 + 13) * RANK + rr, v13)
            tl.store(out_ptr + (tok0 + 14) * RANK + rr, v14)
            tl.store(out_ptr + (tok0 + 15) * RANK + rr, v15)
            tl.store(out_ptr + (tok0 + 16) * RANK + rr, v16)
            tl.store(out_ptr + (tok0 + 17) * RANK + rr, v17)
            tl.store(out_ptr + (tok0 + 18) * RANK + rr, v18)
            tl.store(out_ptr + (tok0 + 19) * RANK + rr, v19)
            tl.store(out_ptr + (tok0 + 20) * RANK + rr, v20)
            tl.store(out_ptr + (tok0 + 21) * RANK + rr, v21)
            tl.store(out_ptr + (tok0 + 22) * RANK + rr, v22)
            tl.store(out_ptr + (tok0 + 23) * RANK + rr, v23)
            tl.store(out_ptr + (tok0 + 24) * RANK + rr, v24)
            tl.store(out_ptr + (tok0 + 25) * RANK + rr, v25)
            tl.store(out_ptr + (tok0 + 26) * RANK + rr, v26)
            tl.store(out_ptr + (tok0 + 27) * RANK + rr, v27)
            tl.store(out_ptr + (tok0 + 28) * RANK + rr, v28)
            tl.store(out_ptr + (tok0 + 29) * RANK + rr, v29)
            tl.store(out_ptr + (tok0 + 30) * RANK + rr, v30)
            tl.store(out_ptr + (tok0 + 31) * RANK + rr, v31)
            tl.store(out_ptr + (tok0 + 32) * RANK + rr, v32)
            tl.store(out_ptr + (tok0 + 33) * RANK + rr, v33)
            tl.store(out_ptr + (tok0 + 34) * RANK + rr, v34)
            tl.store(out_ptr + (tok0 + 35) * RANK + rr, v35)
            tl.store(out_ptr + (tok0 + 36) * RANK + rr, v36)
            tl.store(out_ptr + (tok0 + 37) * RANK + rr, v37)
            tl.store(out_ptr + (tok0 + 38) * RANK + rr, v38)
            tl.store(out_ptr + (tok0 + 39) * RANK + rr, v39)
            tl.store(out_ptr + (tok0 + 40) * RANK + rr, v40)
            tl.store(out_ptr + (tok0 + 41) * RANK + rr, v41)
            tl.store(out_ptr + (tok0 + 42) * RANK + rr, v42)
            tl.store(out_ptr + (tok0 + 43) * RANK + rr, v43)
            tl.store(out_ptr + (tok0 + 44) * RANK + rr, v44)
            tl.store(out_ptr + (tok0 + 45) * RANK + rr, v45)
            tl.store(out_ptr + (tok0 + 46) * RANK + rr, v46)
            tl.store(out_ptr + (tok0 + 47) * RANK + rr, v47)
            tl.store(out_ptr + (tok0 + 48) * RANK + rr, v48)
            tl.store(out_ptr + (tok0 + 49) * RANK + rr, v49)
            tl.store(out_ptr + (tok0 + 50) * RANK + rr, v50)
            tl.store(out_ptr + (tok0 + 51) * RANK + rr, v51)
            tl.store(out_ptr + (tok0 + 52) * RANK + rr, v52)
            tl.store(out_ptr + (tok0 + 53) * RANK + rr, v53)
            tl.store(out_ptr + (tok0 + 54) * RANK + rr, v54)
            tl.store(out_ptr + (tok0 + 55) * RANK + rr, v55)
            tl.store(out_ptr + (tok0 + 56) * RANK + rr, v56)
            tl.store(out_ptr + (tok0 + 57) * RANK + rr, v57)
            tl.store(out_ptr + (tok0 + 58) * RANK + rr, v58)
            tl.store(out_ptr + (tok0 + 59) * RANK + rr, v59)
            tl.store(out_ptr + (tok0 + 60) * RANK + rr, v60)
            tl.store(out_ptr + (tok0 + 61) * RANK + rr, v61)
            tl.store(out_ptr + (tok0 + 62) * RANK + rr, v62)
            tl.store(out_ptr + (tok0 + 63) * RANK + rr, v63)
        else:
            for i in tl.static_range(TPP):
                row = tok0 + i
                if row < end:
                    _token_generic(
                        rr, row, ids_ptr, w_ptr, extra_ptr, out_ptr, wi,
                        active_rank, VOCAB, RANK, RANK_ACT, EXTRA_COUNT,
                        HAS_EXTRA, NEED_MASK, IDS_STRIDE,
                    )


_TPP = 64

_IDS_STRIDE_CACHE = {}


def _ids_int32_words(input_ids):
    """Zero-copy int32 word alias for int64 ids.

    Under ``TORCH_GCU_ENABLE_INT64_AND_UINT64=1`` int64 tensors are real
    8-byte storage, so the little-endian low words are the id values
    themselves (< 2^31). The layout is probe-verified once per device per
    process (fake-int64 compact storage or exotic layouts fall back to the
    ``.to(int32)`` cast kernel, which stays bit-correct in every case).
    Returns ``(tensor, ids_stride)``; kernel addresses fold the constexpr
    stride at compile time.
    """
    if input_ids.dtype == torch.int32:
        return input_ids, 1
    if input_ids.dtype != torch.int64 or not input_ids.is_contiguous():
        return input_ids.to(torch.int32), 1
    device = input_ids.device
    stride = _IDS_STRIDE_CACHE.get(device)
    if stride is None:
        probe = torch.arange(4, device=device, dtype=torch.int64)
        words = probe.view(torch.int32)
        if (
            words.numel() == 8
            and bool((words[::2].cpu() == torch.arange(4)).all())
        ):
            stride = 2
        else:
            stride = 0
        _IDS_STRIDE_CACHE[device] = stride
    if stride == 2:
        return input_ids.view(torch.int32), 2
    return input_ids.to(torch.int32), 1


def embedding_lora_a(
    input_ids, weights, batch_info, vocab_size, extra_embeddings=None
):
    token_count = input_ids.shape[0]
    rank_act = weights.shape[1]
    output = torch.empty(
        (token_count, rank_act), dtype=weights.dtype, device=weights.device
    )
    if token_count == 0:
        return output
    input_ids, ids_stride = _ids_int32_words(input_ids)
    rank = triton.next_power_of_2(rank_act)
    need_mask = rank != rank_act
    token_blocks = triton.cdiv(batch_info.max_len, _TPP)
    total = batch_info.bs * token_blocks
    extra_count = 0 if extra_embeddings is None else extra_embeddings.shape[1]
    extra_pointer = weights if extra_embeddings is None else extra_embeddings
    _embedding_lora_a_kernel[(min(total, 65535),)](
        input_ids,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        extra_pointer,
        output,
        token_blocks,
        total,
        vocab_size,
        rank,
        rank_act,
        extra_count,
        extra_embeddings is not None,
        need_mask,
        _TPP,
        ids_stride,
        num_warps=1,
        num_stages=2,
    )
    return output


__all__ = ["embedding_lora_a"]
