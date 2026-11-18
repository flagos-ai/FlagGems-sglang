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

"""Input preparation shared by the Mamba collection workloads."""

import math


def chunk_cumsum_inputs(dt, a, chunk_size):
    # Match the benchmark's Torch chunk_cumsum reference with no bias or softplus.
    batch, seqlen, heads = dt.shape
    chunks = math.ceil(seqlen / chunk_size)
    dt_out = (
        dt.float()
        .clamp(min=0.0)
        .reshape(batch, chunks, chunk_size, heads)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return dt_out, (dt_out * a.float().view(1, heads, 1, 1)).cumsum(dim=-1)
