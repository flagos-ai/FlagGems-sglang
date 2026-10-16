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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl

_MAX_PROGRAMS = 256
_HEAD = 2  # leading size words when the header shares the device staging area
_MAX_SPIN = (
    20_000_000  # doorbell poll bound before falling back to a sync wait
)


@triton.jit
def _pack_kernel(
    ocl_ptr,  # out_cache_loc (read as a flat (bs * num_verify_tokens) buffer)
    n_ptr,  # (bs,) num_correct_drafts (int)
    f_ptr,  # (bs,) to_free_num_slots (int)
    pay_ptr,  # staging: [tgt_payload, free_payload]
    hdr_ptr,  # (2,) output sizes; may be device *or* pinned host memory
    num_verify_tokens: tl.constexpr,
    bs: tl.constexpr,
    rows_per_prog: tl.constexpr,
    BLOCK_BS: tl.constexpr,  # >= bs, for the leading prefix sums
    BLOCK_N: tl.constexpr,  # >= num_verify_tokens
):
    pid = tl.program_id(0)
    b0 = pid * rows_per_prog
    span: tl.constexpr = bs * num_verify_tokens  # per-output payload capacity

    bofs = tl.arange(0, BLOCK_BS)

    if pid == 0:
        total_n = tl.sum(
            tl.load(n_ptr + bofs, mask=bofs < bs, other=0).to(tl.int64)
        )
        total_f = tl.sum(
            tl.load(f_ptr + bofs, mask=bofs < bs, other=0).to(tl.int64)
        )
        tl.store(hdr_ptr + 0, total_n + bs)
        tl.store(hdr_ptr + 1, total_f)

    # Running (exclusive) prefix sums of the two count vectors, seeded with
    # everything strictly before this program's first row.
    tgt_off = (
        tl.sum(tl.load(n_ptr + bofs, mask=bofs < b0, other=0).to(tl.int64))
        + b0
    )
    free_off = tl.sum(
        tl.load(f_ptr + bofs, mask=bofs < b0, other=0).to(tl.int64)
    )

    idx = tl.arange(0, BLOCK_N)

    for r in tl.static_range(0, rows_per_prog):
        b = b0 + r
        if b < bs:
            n_b = tl.load(n_ptr + b).to(tl.int64)
            f_b = tl.load(f_ptr + b).to(tl.int64)
            keep = n_b + 1

            # Whole verify row loaded once, scattered to both destinations.
            vals = tl.load(
                ocl_ptr + b * num_verify_tokens + idx,
                mask=idx < num_verify_tokens,
                other=0,
            )
            tl.store(pay_ptr + tgt_off + idx, vals, mask=idx < keep)

            free_src = num_verify_tokens - f_b
            tl.store(
                pay_ptr + span + free_off + (idx - free_src),
                vals,
                mask=(idx >= free_src) & (idx < num_verify_tokens),
            )

            tgt_off += keep
            free_off += f_b


def _pinned_header(device):
    """Allocate the 2-word size header in pinned memory when supported."""
    try:
        return torch.empty(_HEAD, dtype=torch.int64, pin_memory=True)
    except (RuntimeError, TypeError, AssertionError, NotImplementedError):
        return torch.empty(_HEAD, dtype=torch.int64, device=device)


def get_target_cache_loc(
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    bs = num_correct_drafts.shape[0]
    device = out_cache_loc.device
    dtype = out_cache_loc.dtype

    if bs == 0:
        # Sizes only, matching the reference's empty-batch result.
        return torch.empty(1, dtype=dtype, device=device), torch.empty(
            1, dtype=dtype, device=device
        )

    n_prog = min(bs, _MAX_PROGRAMS)
    rows_per_prog = -(-bs // n_prog)
    span = max(bs * num_verify_tokens, 1)  # per-output speculative capacity

    # One staging allocation holds both packed payloads; each finished output
    # is a view of this buffer, so nothing is copied on the host.
    stage = torch.empty(2 * span, dtype=dtype, device=device)
    header = _pinned_header(device)

    doorbell = header.is_pinned()
    if doorbell:
        header.fill_(-1)

    _pack_kernel[(n_prog,)](
        out_cache_loc,
        num_correct_drafts,
        to_free_num_slots,
        stage,
        header,
        num_verify_tokens=num_verify_tokens,
        bs=bs,
        rows_per_prog=rows_per_prog,
        BLOCK_BS=triton.next_power_of_2(bs),
        BLOCK_N=triton.next_power_of_2(num_verify_tokens),
        num_warps=1,
        num_stages=1,
    )

    if doorbell:
        for _ in range(_MAX_SPIN):
            if header.tolist()[0] != -1:
                break
        else:
            torch.cuda.current_stream().synchronize()
    else:
        # No pinned host memory on this backend: the header lives on the
        # device, so an event wait before the scalar read is the whole tail.
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()

    n_tgt, n_free = header.tolist()

    tgt = stage[0:n_tgt]
    free = stage[span : span + n_free]
    if n_free == 0:
        free = torch.zeros(1, dtype=dtype, device=device)
    return tgt, free


__all__ = ["get_target_cache_loc"]
