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


@triton.jit
def _decode(raw):
    q = raw.to(tl.uint8).to(tl.int8, bitcast=True).to(tl.int16)
    bits = (q << 7).to(tl.uint16) & 49151
    value = bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return tl.where(q & 127 == 127, float("nan"), value)


@triton.jit
def _kernel(X, Y, S: tl.constexpr, G: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, 512)
    q = tl.load(X + row * S + c).to(tl.uint32)
    scales = (X + row * S + 512).to(tl.pointer_type(tl.float32))
    s = tl.load(scales + c // G)
    r = tl.arange(0, 64)
    rope = (X + row * S + 512 + 512 // G * 4).to(tl.pointer_type(tl.uint16))
    bits = tl.load(rope + r)
    tl.store(Y + row * 576 + c, _decode(q) * s)
    out = Y.to(tl.pointer_type(tl.uint16))
    tl.store(out + row * 576 + 512 + r, bits)


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    n = quant_k_cache.shape[0]
    output = torch.empty(
        (n, 576), dtype=torch.bfloat16, device=quant_k_cache.device
    )
    if n:
        _kernel.run(
            quant_k_cache.view(torch.uint8),
            output,
            quant_k_cache.stride(0),
            group_size,
            grid=(n,),
            warmup=False,
            num_warps=2,
        )
    return output


__all__ = ["dequantize_k_cache_fast"]
