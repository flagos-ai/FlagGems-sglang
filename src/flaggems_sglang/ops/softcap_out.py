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

import torch
import triton
import triton.language as tl


# Keep the one-dimensional physical launch below known heterogeneous-backend
# limits.  Large tensors continue through a short grid-stride loop.
_MAX_PROGRAMS = 32768


@triton.jit
def _softcap_out_kernel(
    output_ptr,
    input_ptr,
    softcap_const: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    physical_pid = tl.program_id(0)

    for block_id in tl.range(
        physical_pid,
        N_BLOCKS,
        NUM_PROGRAMS,
    ):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = x / softcap_const

        # tanh(z) = sign(z) * (1 - exp(-2*abs(z))) /
        #                         (1 + exp(-2*abs(z))).
        # Unlike (exp(2*z)-1)/(exp(2*z)+1), the exponential argument is never
        # positive, so large positive logits cannot produce inf/inf -> NaN.
        abs_z = tl.abs(z)
        exp_neg = tl.exp(-2.0 * abs_z)
        magnitude = (1.0 - exp_neg) / (1.0 + exp_neg)
        tanh_z = tl.where(z < 0.0, -magnitude, magnitude)
        result = tanh_z * softcap_const

        tl.store(output_ptr + offsets, result, mask=mask)


def _select_launch_config(n_elements: int):
    # Small logits benefit from more independent programs; larger tensors use a
    # wider vector tile to amortize indexing and loop overhead while retaining
    # enough parallelism to cover tanh/exp latency.
    if n_elements <= 128:
        return 128, 4
    if n_elements < 4096:
        return 256, 4
    return 512, 4


def softcap_out(x, softcap_const, autotune=False):
    """Apply ``softcap * tanh(x / softcap)`` and return an FP32 tensor.

    The ``autotune`` argument is retained for exact SGLang API compatibility.
    A deterministic shape heuristic is used on every backend to avoid expensive
    first-call benchmarking and unsupported configurations on non-CUDA chips.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    block_size, num_warps = _select_launch_config(n_elements)
    n_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(n_blocks, _MAX_PROGRAMS)

    _softcap_out_kernel[(num_programs,)](
        output,
        x,
        softcap_const=softcap_const,
        N_ELEMENTS=n_elements,
        N_BLOCKS=n_blocks,
        NUM_PROGRAMS=num_programs,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


__all__ = ["softcap_out"]
