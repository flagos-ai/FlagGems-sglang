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
"""Build permutation indices for masked/DeepGEMM MoE dispatch.
"""

import functools

import torch
import triton
import triton.language as tl

import flaggems_sglang

# Kernel launch configuration, tuned on the metax backend (the multi-program
# BLOCK=256/num_warps=4 point matches the vendor default in
# runtime/backend/_metax/tune_configs.yaml and measured fastest across the
# bench shapes; the single-program point was tuned by sweep).
_BLOCK = 256
_NUM_WARPS = 4
# numel up to this value is processed by the single-program kernel (its
# in-kernel cursor zeroing saves the second launch, which dominates below the
# ~12us launch-latency floor); above it the parallel multi-program kernel's
# overlap wins.  Measured crossover under the benchmark harness is between
# 1024 and 2048.
_SINGLE_MAX_NUMEL = 1024
_SINGLE_BLOCK = 512
_SINGLE_WARPS = 8


@triton.jit(do_not_specialize=["numel", "m_max", "num_experts"])
def _dispatch_index_single_kernel(
    ids_ptr,  # [numel] int32 expert ids (-1 = padding)
    mm_ptr,  # [num_local_experts] int32 cursors; zeroed in-kernel
    dst_ptr,  # [numel] int32 destination rows
    numel,
    m_max,
    num_experts,
    BLOCK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # Zero the cursors without a host-side memset: the whole tensor is owned
    # by this single program, so a CTA barrier orders the stores before the
    # atomic hand-out below.
    e_offs = tl.arange(0, BLOCK_E)
    tl.store(
        mm_ptr + e_offs,
        tl.zeros((BLOCK_E,), tl.int32),
        mask=e_offs < num_experts,
    )
    tl.debug_barrier()

    for start in range(0, numel, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        in_bounds = offs < numel
        e = tl.load(ids_ptr + offs, mask=in_bounds, other=-1)
        valid = in_bounds & (e >= 0)
        e_safe = tl.where(valid, e, 0)
        # Relaxed ordering suffices: we only need the atomic RMW to hand out
        # unique offsets, no cross-thread visibility guarantees.
        slot = tl.atomic_add(mm_ptr + e_safe, 1, mask=valid, sem="relaxed")
        dst = e_safe * m_max + slot
        dst = tl.where(valid, dst, 0)
        tl.store(dst_ptr + offs, dst, mask=in_bounds)


@triton.jit(do_not_specialize=["num_experts"])
def _zero_cursors_kernel(
    mm_ptr,  # [num_local_experts] int32 cursors
    num_experts,
    BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    tl.store(
        mm_ptr + offs, tl.zeros((BLOCK_E,), tl.int32), mask=offs < num_experts
    )


@triton.jit(do_not_specialize=["numel", "m_max"])
def _dispatch_index_multi_kernel(
    ids_ptr,  # [numel] int32 expert ids (-1 = padding)
    mm_ptr,  # [num_local_experts] int32 cursors; zeroed beforehand
    dst_ptr,  # [numel] int32 destination rows
    numel,
    m_max,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        e = tl.load(ids_ptr + offs)
        valid = e >= 0
    else:
        in_bounds = offs < numel
        e = tl.load(ids_ptr + offs, mask=in_bounds, other=-1)
        valid = in_bounds & (e >= 0)

    # Clamp invalid lanes to a safe (but masked-out) expert id so the atomic
    # below never sees a negative offset.
    e_safe = tl.where(valid, e, 0)

    # Relaxed ordering suffices: we only need the atomic RMW to hand out
    # unique offsets, no cross-thread visibility guarantees.
    slot = tl.atomic_add(mm_ptr + e_safe, 1, mask=valid, sem="relaxed")

    dst = e_safe * m_max + slot
    dst = tl.where(valid, dst, 0)
    if EVEN:
        tl.store(dst_ptr + offs, dst)
    else:
        tl.store(dst_ptr + offs, dst, mask=offs < numel)


def _driver():
    return triton.runtime.driver.active


def _raw_get_device():
    """Prefer the raw C device getter over the driver's Python wrapper.

    Adopted only when it exists and agrees with the driver's own query for the
    active device; otherwise the driver query is kept so other backends stay
    correct.
    """
    drv = _driver()
    get_device = drv.get_current_device
    try:
        raw = torch._C._cuda_getDevice
        if raw() == get_device():
            return raw
    except Exception:
        pass
    return get_device


def _raw_get_stream(get_device):
    """Prefer the raw C stream getter (the one torch inductor launches with).

    Adopted only when it exists and returns the same raw handle the driver's
    own query reports; otherwise the driver query is kept.
    """
    drv = _driver()
    get_stream = drv.get_current_stream
    try:
        raw = torch._C._cuda_getCurrentRawStream
        if raw(get_device()) == get_stream(get_device()):
            return raw
    except Exception:
        pass
    return get_stream


def _resolve_launch(ko, probe_args):
    """Probe the raw launch entry point of a compiled kernel.

    Tries the two known launcher layouts (with and without the global /
    profile scratch slots some backends insert before the metadata argument);
    returns ``(launch, has_scratch)`` or ``None`` if neither matches, in which
    case the caller falls back to the regular JIT dispatch.
    """
    launch = ko.run.launch
    get_device = _raw_get_device()
    get_stream = _raw_get_stream(get_device)
    stream = get_stream(get_device())
    # Layout A: (gridX, gridY, gridZ, stream, function, global_scratch,
    #            profile_scratch, kernel_metadata, launch_metadata,
    #            enter_hook, exit_hook, *args)
    try:
        launch(
            1,
            1,
            1,
            stream,
            ko.function,
            None,
            None,
            ko.packed_metadata,
            None,
            None,
            None,
            *probe_args,
        )
        return launch, True
    except TypeError:
        pass
    # Layout B: (gridX, gridY, gridZ, stream, function, kernel_metadata,
    #            launch_metadata, enter_hook, exit_hook, *args)
    try:
        launch(
            1,
            1,
            1,
            stream,
            ko.function,
            ko.packed_metadata,
            None,
            None,
            None,
            *probe_args,
        )
        return launch, False
    except TypeError:
        return None


@functools.lru_cache(maxsize=16)
def _fast_single_cached(num_local_experts, m_max):
    """Compile the single-program variant; return its raw-launch closure.

    Only compilation artifacts are cached (the CompiledKernel, its raw launch
    entry point, packed metadata and stream/device getters — no tensors, no
    results), mirroring ``create_flashinfer_kv_indices._get_fast_call``.  The
    probe launch runs on zero-filled dummies sized to the block, so it never
    reads or writes beyond them.  Returns ``None`` when the raw launch
    signature does not match, in which case the regular JIT dispatch is used.
    """
    device = flaggems_sglang.device
    block = _SINGLE_BLOCK
    block_e = max(triton.next_power_of_2(num_local_experts), 2)
    ph = torch.zeros(block, dtype=torch.int32, device=device)
    ph_mm = torch.zeros(block_e, dtype=torch.int32, device=device)
    ko = _dispatch_index_single_kernel.warmup(
        ph,
        ph_mm,
        ph,
        block,
        m_max,
        num_local_experts,
        BLOCK=block,
        BLOCK_E=block_e,
        num_warps=_SINGLE_WARPS,
        grid=(1,),
    )
    probe_args = (
        ph.data_ptr(),
        ph_mm.data_ptr(),
        ph.data_ptr(),
        block,
        m_max,
        num_local_experts,
        0,
        0,
    )
    resolved = _resolve_launch(ko, probe_args)

    if resolved is None:
        return None
    if resolved[1]:

        def fast_call(
            ids,
            mm,
            dst,
            numel,
            ne,
            mx,
            _launch=resolved[0],
            _fn=ko.function,
            _km=ko.packed_metadata,
            _gs=_raw_get_stream(_raw_get_device()),
            _gd=_raw_get_device(),
        ):
            _launch(
                1,
                1,
                1,
                _gs(_gd()),
                _fn,
                None,
                None,
                _km,
                None,
                None,
                None,
                ids.data_ptr(),
                mm.data_ptr(),
                dst.data_ptr(),
                numel,
                mx,
                ne,
                0,
                0,
            )
            return dst

    else:

        def fast_call(
            ids,
            mm,
            dst,
            numel,
            ne,
            mx,
            _launch=resolved[0],
            _fn=ko.function,
            _km=ko.packed_metadata,
            _gs=_raw_get_stream(_raw_get_device()),
            _gd=_raw_get_device(),
        ):
            _launch(
                1,
                1,
                1,
                _gs(_gd()),
                _fn,
                _km,
                None,
                None,
                None,
                ids.data_ptr(),
                mm.data_ptr(),
                dst.data_ptr(),
                numel,
                mx,
                ne,
                0,
                0,
            )
            return dst

    return fast_call


@functools.lru_cache(maxsize=16)
def _fast_zero_cached(num_local_experts):
    """Compile the cursor-zeroing kernel; return its raw-launch closure."""
    device = flaggems_sglang.device
    block_e = max(triton.next_power_of_2(num_local_experts), 2)
    ph_mm = torch.zeros(block_e, dtype=torch.int32, device=device)
    ko = _zero_cursors_kernel.warmup(
        ph_mm,
        num_local_experts,
        BLOCK_E=block_e,
        num_warps=1,
        grid=(1,),
    )
    # Trailing slot count must match the launcher's argument format, which
    # covers every signature entry including the constexpr placeholder.
    probe_args = (ph_mm.data_ptr(), num_local_experts, 0)
    resolved = _resolve_launch(ko, probe_args)

    if resolved is None:
        return None
    if resolved[1]:

        def fast_zero(
            mm,
            ne,
            _launch=resolved[0],
            _fn=ko.function,
            _km=ko.packed_metadata,
            _gs=_raw_get_stream(_raw_get_device()),
            _gd=_raw_get_device(),
        ):
            _launch(
                1,
                1,
                1,
                _gs(_gd()),
                _fn,
                None,
                None,
                _km,
                None,
                None,
                None,
                mm.data_ptr(),
                ne,
                0,
            )
            return mm

    else:

        def fast_zero(
            mm,
            ne,
            _launch=resolved[0],
            _fn=ko.function,
            _km=ko.packed_metadata,
            _gs=_raw_get_stream(_raw_get_device()),
            _gd=_raw_get_device(),
        ):
            _launch(
                1,
                1,
                1,
                _gs(_gd()),
                _fn,
                _km,
                None,
                None,
                None,
                mm.data_ptr(),
                ne,
                0,
            )
            return mm

    return fast_zero


@functools.lru_cache(maxsize=16)
def _fast_multi_cached(num_local_experts, m_max, even):
    """Compile the multi-program EVEN-variant; return its raw-launch closure.

    Only compilation artifacts are cached (no tensors, no results); the probe
    launch runs on zero-filled dummies sized to the block.  Returns ``None``
    when the raw launch signature does not match, in which case the regular
    JIT dispatch is used.
    """
    device = flaggems_sglang.device
    ph_ids = torch.zeros(_BLOCK, dtype=torch.int32, device=device)
    ph_mm = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    ko = _dispatch_index_multi_kernel.warmup(
        ph_ids,
        ph_mm,
        ph_ids,
        _BLOCK,
        m_max,
        BLOCK=_BLOCK,
        EVEN=even,
        num_warps=_NUM_WARPS,
        grid=(1,),
    )
    probe_args = (
        ph_ids.data_ptr(),
        ph_mm.data_ptr(),
        ph_ids.data_ptr(),
        _BLOCK,
        m_max,
        0,
        0,
    )
    resolved = _resolve_launch(ko, probe_args)

    if resolved is None:
        return None
    if resolved[1]:

        def fast_call(
            ids,
            mm,
            dst,
            numel,
            grid_x,
            mx,
            _launch=resolved[0],
            _fn=ko.function,
            _km=ko.packed_metadata,
            _gs=_raw_get_stream(_raw_get_device()),
            _gd=_raw_get_device(),
        ):
            _launch(
                grid_x,
                1,
                1,
                _gs(_gd()),
                _fn,
                None,
                None,
                _km,
                None,
                None,
                None,
                ids.data_ptr(),
                mm.data_ptr(),
                dst.data_ptr(),
                numel,
                mx,
                0,
                0,
            )
            return dst

    else:

        def fast_call(
            ids,
            mm,
            dst,
            numel,
            grid_x,
            mx,
            _launch=resolved[0],
            _fn=ko.function,
            _km=ko.packed_metadata,
            _gs=_raw_get_stream(_raw_get_device()),
            _gd=_raw_get_device(),
        ):
            _launch(
                grid_x,
                1,
                1,
                _gs(_gd()),
                _fn,
                _km,
                None,
                None,
                None,
                ids.data_ptr(),
                mm.data_ptr(),
                dst.data_ptr(),
                numel,
                mx,
                0,
                0,
            )
            return dst

    return fast_call


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    # The kernels only need a flat pointer + element count; pass the tensor
    # through as-is when contiguous so a no-op ``reshape`` does not pay a host
    # dispatch on every call.
    if topk_ids.is_contiguous():
        flat = topk_ids
    else:
        flat = topk_ids.reshape(-1)
    numel = flat.numel()
    if numel == 0:
        return (
            torch.zeros(
                num_local_experts, dtype=torch.int32, device=flat.device
            ),
            torch.empty(0, dtype=torch.int32, device=flat.device),
        )

    device = flat.device
    if numel <= _SINGLE_MAX_NUMEL:
        # Small workloads: one program owns the whole tensor, so the cursor
        # zeroing is folded into the kernel and no separate memset dispatch
        # is needed.  The two outputs are plain allocations — no fused buffer,
        # no slice views on the hot path.
        masked_m = torch.empty(
            num_local_experts, dtype=torch.int32, device=device
        )
        src2dst = torch.empty(numel, dtype=torch.int32, device=device)
        fast = _fast_single_cached(num_local_experts, m_max)
        if fast is not None:
            fast(flat, masked_m, src2dst, numel, num_local_experts, m_max)
        else:
            _dispatch_index_single_kernel[(1,)](
                flat,
                masked_m,
                src2dst,
                numel,
                m_max,
                num_local_experts,
                BLOCK=_SINGLE_BLOCK,
                BLOCK_E=max(triton.next_power_of_2(num_local_experts), 2),
                num_warps=_SINGLE_WARPS,
            )
        return masked_m, src2dst

    # Large workloads: zero the cursors with a dedicated one-program kernel
    # (raw C launch, cheaper than a torch.zeros allocate+memset dispatch),
    # then run the parallel multi-program kernel.  Same-stream ordering
    # guarantees the zeroing lands before the atomic hand-outs.
    masked_m = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(numel, dtype=torch.int32, device=device)
    fast_zero = _fast_zero_cached(num_local_experts)
    if fast_zero is not None:
        fast_zero(masked_m, num_local_experts)
    else:
        _zero_cursors_kernel[(1,)](
            masked_m,
            num_local_experts,
            BLOCK_E=max(triton.next_power_of_2(num_local_experts), 2),
            num_warps=1,
        )
    even = numel % _BLOCK == 0
    fast = _fast_multi_cached(num_local_experts, m_max, even)
    if fast is not None:
        fast(
            flat,
            masked_m,
            src2dst,
            numel,
            (numel + _BLOCK - 1) // _BLOCK,
            m_max,
        )
    else:
        _dispatch_index_multi_kernel[((numel + _BLOCK - 1) // _BLOCK,)](
            flat,
            masked_m,
            src2dst,
            numel,
            m_max,
            BLOCK=_BLOCK,
            EVEN=even,
            num_warps=_NUM_WARPS,
        )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
