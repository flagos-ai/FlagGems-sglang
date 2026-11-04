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

"""LoRA 系列算子共用的 LoRABatchInfo 构造（复刻
problems/lora/_batch_utils.py 的 make_batch_info）。

采集器把 ops/<repo> 目录加入 sys.path 后按模块名 import 各算子模块；本模块
作为同目录 sibling 供 LoRA 算子模块 `from _lora_batch_utils import
make_batch_info` 引用。LoRABatchInfo 在函数内运行时 import（sglang 未安装时
不影响模块加载，与各算子 native() 的惰性解析一致）。
"""

import torch


def make_batch_info(seg_lens, weight_indices, lora_ranks, device="cuda",
                    scalings=None, permutation="none"):
    """构造 sglang.srt.lora.utils.LoRABatchInfo。

    permutation："none"（SORTED_BY_ADAPTER=False，段内 token 连续）、
    "identity"（保序显式置换）、"shuffled"（真实乱序）。
    """
    from sglang.srt.lora.utils import LoRABatchInfo

    bs = len(seg_lens)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, dim=0)
    weight_indices_t = torch.tensor(weight_indices, dtype=torch.int32, device=device)
    lora_ranks_t = torch.tensor(lora_ranks, dtype=torch.int32, device=device)
    if scalings is None:
        scalings = [1.0] * len(lora_ranks)
    scalings_t = torch.tensor(scalings, dtype=torch.float32, device=device)
    max_len = max(seg_lens) if seg_lens else 0
    total_tokens = int(seg_indptr[-1].item())

    if permutation == "none":
        permutation_t = None
    elif permutation == "identity":
        permutation_t = torch.arange(total_tokens, device=device, dtype=torch.int32)
    elif permutation == "shuffled":
        permutation_t = torch.randperm(total_tokens, device=device).to(torch.int32)
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
