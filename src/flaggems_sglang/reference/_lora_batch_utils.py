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


"""Shared ``LoRABatchInfo`` construction for LoRA problems' ``cases.py``.

Not part of any solution's contract -- just test-case scaffolding shared
across the LoRA problem folders, all of which take the same
segment/adapter-routing metadata.
"""

import torch

import flaggems_sglang

try:
    from sglang.srt.lora.utils import LoRABatchInfo
except ImportError:
    from flaggems_reference._lora_batch_info import LoRABatchInfo


def make_batch_info(
    seg_lens, weight_indices, lora_ranks, scalings=None, permutation="none"
):
    """``permutation``: "none" (SORTED_BY_ADAPTER=False; tokens contiguous per
    segment), "identity" (an explicit but order-preserving permutation, for
    kernels that always require one), or "shuffled" (exercises real
    re-ordering)."""
    device = flaggems_sglang.device
    bs = len(seg_lens)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, dim=0)
    weight_indices_t = torch.tensor(
        weight_indices, dtype=torch.int32, device=device
    )
    lora_ranks_t = torch.tensor(lora_ranks, dtype=torch.int32, device=device)
    if scalings is None:
        scalings = [1.0] * len(lora_ranks)
    scalings_t = torch.tensor(scalings, dtype=torch.float32, device=device)
    max_len = max(seg_lens) if seg_lens else 0
    total_tokens = int(seg_indptr[-1].item())

    if permutation == "none":
        permutation_t = None
    elif permutation == "identity":
        permutation_t = torch.arange(
            total_tokens, device=device, dtype=torch.int32
        )
    elif permutation == "shuffled":
        # torch_gcu randperm hangs on-device; generate on CPU then move.
        permutation_t = (
            torch.randperm(total_tokens, device="cpu")
            .to(torch.int32)
            .to(device)
        )
    else:
        raise ValueError(permutation)

    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices_t,
        lora_ranks=lora_ranks_t,
        scalings=scalings_t,
        max_len=max_len,
        seg_lens=seg_lens_t,
        permutation=permutation_t,
    )
