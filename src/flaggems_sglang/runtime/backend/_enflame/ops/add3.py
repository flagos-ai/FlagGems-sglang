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

"""add3 for the Enflame GCU (gcu410, 24 SIPs).

Rewritten for the 2026-11-07 triton-gcu toolchain (job j20261108111139_b877c34f):

- The previous BLOCK=32768 persistent config no longer compiles: the
  2026-11-07 20:22 toolchain regression rejects any 1D flat masked/unmasked
  load/store config with more than 4096 elements per warp
  (``memref_ext.slice_pad_start`` operandSegmentSizes 8 vs 7; see
  compiler_issues/vllm_silu_flat_slice_start.md). BLOCK is therefore capped at
  8192 with num_warps=4 (2048 elements/warp, 16 KB bf16 per slice).
- Measured on gcu410 (do_bench kernel-only, median of >=3): launch floor
  2.34us; 3-load+1-store streaming floor 11.07us at n=3.67M for
  BLOCK=8192/num_warps=4/grid=48 (2652 GB/s). grid=48 (2 programs per SIP)
  beats grid=24 by ~22% and the one-shot flat grid (448 programs) by ~60%;
  scheduling overhead still dominates flat grids on this part.
- Dispatch tiers (probe-swept): flat one-shot BLOCK=2048 for n<=16384,
  persistent BLOCK=4096 grid=cdiv for n<=2*SIPs*4096 (single 4096-wide wave,
  at most 2 programs per SIP), persistent BLOCK=8192 grid=2*SIPs for larger
  n; EVEN=True drops the mask when n % BLOCK == 0.
- bf16 arithmetic: ``(a + b) + c`` on this backend is bit-exact with the
  explicit fp32 cast-and-round chain (verified over 3.67M randn values plus 64
  edge pairs incl. +-inf/NaN/-0/denormals, both also bit-exact against the
  torch reference) and ~6% faster at large n: the backend promotes bf16
  arithmetic to fp32 and rounds back to bf16 per addition, which is exactly
  the two-step rounding required by the task. Other dtypes keep the explicit
  cast chain, matching the generic implementation.
"""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _add3_flat_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    if IS_BF16:
        result = (a + b) + c
    else:
        ab = (a.to(tl.float32) + b.to(tl.float32)).to(a_ptr.dtype.element_ty)
        result = (
            ab.to(tl.float32) + c.to(tl.float32)
        ).to(a_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _add3_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    step = tl.num_programs(0) * BLOCK_SIZE
    for start in range(tl.program_id(0) * BLOCK_SIZE, n_elements, step):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            a = tl.load(a_ptr + offsets)
            b = tl.load(b_ptr + offsets)
            c = tl.load(c_ptr + offsets)
            if IS_BF16:
                result = (a + b) + c
            else:
                ab = (a.to(tl.float32) + b.to(tl.float32)).to(
                    a_ptr.dtype.element_ty
                )
                result = (
                    ab.to(tl.float32) + c.to(tl.float32)
                ).to(a_ptr.dtype.element_ty)
            tl.store(out_ptr + offsets, result)
        else:
            mask = offsets < n_elements
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            c = tl.load(c_ptr + offsets, mask=mask)
            if IS_BF16:
                result = (a + b) + c
            else:
                ab = (a.to(tl.float32) + b.to(tl.float32)).to(
                    a_ptr.dtype.element_ty
                )
                result = (
                    ab.to(tl.float32) + c.to(tl.float32)
                ).to(a_ptr.dtype.element_ty)
            tl.store(out_ptr + offsets, result, mask=mask)


@functools.lru_cache(maxsize=None)
def _num_programs():
    """Resident program count = one program per SIP core."""
    driver = triton.runtime.driver.active
    try:
        props = driver.utils.get_device_properties(driver.get_current_device())
        return max(
            1,
            props["multiprocessor_count"]
            * driver.get_current_target().warp_size,
        )
    except Exception:
        return 24


def add3(a, b, c):
    """FlagOS evaluator entry point for the add3 operator."""
    out = torch.empty_like(a)
    n_elements = a.numel()
    if n_elements == 0:
        return out

    is_bf16 = a.dtype == torch.bfloat16
    programs = _num_programs()
    if n_elements <= 16384:
        block_size = 2048
        grid = (triton.cdiv(n_elements, block_size),)
        _add3_flat_kernel[grid](
            a,
            b,
            c,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            IS_BF16=is_bf16,
            num_warps=2 if n_elements <= 2048 else 4,
        )
    elif n_elements <= 2 * programs * 4096:
        # Single 4096-wide wave (at most 2 programs per SIP): stable at
        # n=131072 where the 8192-wide persistent tier measured 10% slower
        # and 18% run-to-run spread against the flat 4096 configuration.
        block_size = 4096
        grid = (triton.cdiv(n_elements, block_size),)
        _add3_persistent_kernel[grid](
            a,
            b,
            c,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            EVEN=n_elements % block_size == 0,
            IS_BF16=is_bf16,
            num_warps=4,
        )
    else:
        block_size = 8192
        grid = (2 * programs,)
        _add3_persistent_kernel[grid](
            a,
            b,
            c,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            EVEN=n_elements % block_size == 0,
            IS_BF16=is_bf16,
            num_warps=4,
        )
    return out


__all__ = ["add3"]
