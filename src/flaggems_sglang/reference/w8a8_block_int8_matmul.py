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

import torch


def reference(A, B, As, Bs, block_size, output_dtype):
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    block_n, block_k = block_size
    M, K = A.shape
    N, _ = B.shape

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    for i in range(k_tiles):
        k_lo, k_hi = i * block_k, min((i + 1) * block_k, K)
        a_tile = A[:, k_lo:k_hi]
        a_s = As[:, i : i + 1]
        for j in range(n_tiles):
            n_lo, n_hi = j * block_n, min((j + 1) * block_n, N)
            b_tile = B[n_lo:n_hi, k_lo:k_hi]
            s = a_s * Bs[j, i]
            C[:, n_lo:n_hi] += torch.matmul(a_tile, b_tile.t()) * s

    return C.to(output_dtype)
