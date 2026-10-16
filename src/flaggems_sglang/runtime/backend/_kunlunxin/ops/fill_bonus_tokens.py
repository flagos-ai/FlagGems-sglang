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
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=["src", "lens", "out"])
def _gather(
    src, lens, out, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    length = tl.load(lens + row, row < N, 1)
    if N * STRIDE < 2147483648:
        index = row * STRIDE + length.to(tl.int32) - 1
    else:
        index = row.to(tl.int64) * STRIDE + length.to(tl.int64) - 1
    value = tl.load(src + index, row < N, 0)
    tl.store(out + row, value, row < N)


@triton.jit(do_not_specialize_on_alignment=["src", "lens", "out"])
def _full(
    src, lens, out, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    length = tl.load(lens + row).to(tl.int32)
    value = tl.load(src + row * STRIDE + length - 1)
    tl.store(out + row, value)


@tle.raw.dialect(
    "xpu3",
    source=(
        "\n"
        "typedef long long I64;\n"
        "typedef I64 __attribute__((address_space(1))) GI64;\n"
        'extern "C" __attribute__((device)) void raw_gather(const GI64 *src,\n'
        "                                                   const GI64 *lens,\n"
        "                                                   GI64 *out,\n"
        "                                                   int n, int stride, int base, int block) {\n"
        "    I64 lengths[4];\n"
        "    I64 tokens[512];\n"
        "    I64 result[4];\n"
        "    int limit = base + block < n ? base + block : n;\n"
        "    int core = __builtin_xpu_core_id();\n"
        "    for (int row = base + core * 4; row < limit; row += 64 * 4) {\n"
        "        int count = limit - row < 4 ? limit - row : 4;\n"
        "        __builtin_xpu_gm2lm_v3(lengths, lens + row, 0, count * 8);\n"
        "        __builtin_xpu_gm2lm_v3(tokens, src + (I64)row * stride, 0, count * stride * 8);\n"
        "        __builtin_xpu2_mfence(5);\n"
        "        for (int i = 0; i < count; ++i) {\n"
        "            result[i] = tokens[i * stride + lengths[i] - 1];\n"
        "        }\n"
        "        __builtin_xpu_lm2gm_v3(out + row, result, 0, count * 8);\n"
        "        __builtin_xpu2_mfence(5);\n"
        "    }\n"
        "}\n"
    ),
)
def raw_gather(src, lens, out, n, stride, base, block):
    pass


@triton.jit
def _raw(src, lens, out, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    tle.raw.call(
        raw_gather,
        (
            src,
            lens,
            out,
            tl.full((), N, tl.int32),
            tl.full((), S, tl.int32),
            tl.program_id(0) * B,
            tl.full((), B, tl.int32),
        ),
    )


@tle.raw.dialect(
    "xpu3",
    source=(
        "\n"
        "typedef long long I64;\n"
        "typedef I64 __attribute__((address_space(1))) GI64;\n"
        'extern "C" __attribute__((device)) void raw_gather8(const GI64 *src, const GI64 *lens,\n'
        "GI64 *out, int n, int stride, int base, int block) {\n"
        "    I64 lengths[8];\n"
        "    I64 tokens[256];\n"
        "    I64 result[8];\n"
        "    int limit = base + block < n ? base + block : n;\n"
        "    int core = __builtin_xpu_core_id();\n"
        "    for (int row = base + core * 8; row < limit; row += 64 * 8) {\n"
        "        int count = limit - row < 8 ? limit - row : 8;\n"
        "        __builtin_xpu_gm2lm_v3(lengths, lens + row, 0, count * 8);\n"
        "        __builtin_xpu_gm2lm_v3(tokens, src + (I64)row * stride, 0, count * stride * 8);\n"
        "        __builtin_xpu2_mfence(5);\n"
        "        for (int i = 0; i < count; ++i) result[i] = tokens[i * stride + lengths[i] - 1];\n"
        "        __builtin_xpu_lm2gm_v3(out + row, result, 0, count * 8);\n"
        "        __builtin_xpu2_mfence(5);\n"
        "    }\n"
        "}\n"
    ),
)
def raw_gather8(src, lens, out, n, stride, base, block):
    pass


@triton.jit
def _raw8(src, lens, out, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    tle.raw.call(
        raw_gather8,
        (
            src,
            lens,
            out,
            tl.full((), N, tl.int32),
            tl.full((), S, tl.int32),
            tl.program_id(0) * B,
            tl.full((), B, tl.int32),
        ),
    )


def fill_bonus_tokens(accept_tokens, accept_lens, bonus_tokens, accept_stride):
    n = accept_lens.shape[0]
    out = (
        bonus_tokens
        if bonus_tokens.shape == (n,) and bonus_tokens.is_contiguous()
        else torch.empty(
            n, dtype=bonus_tokens.dtype, device=accept_tokens.device
        )
    )
    if n == 0:
        return out
    src = accept_tokens.contiguous()
    lens = accept_lens.contiguous()
    stride = int(accept_stride)
    block = 32 if n <= 32 else 128 if n <= 256 else 256 if n <= 2048 else 512
    full = n >= 256 and n % block == 0 and n * stride < 2147483648
    use_raw = (
        src.dtype == torch.int64
        and lens.dtype == torch.int64
        and out.dtype == torch.int64
        and 0 < stride <= 128
        and n <= 2147483135
    )
    if use_raw:
        block = 256 if n <= 2048 else 512
    kernel = (
        (_raw8 if n > 2048 and stride <= 32 else _raw)
        if use_raw
        else (_full if full else _gather)
    )
    kernel[(triton.cdiv(n, block),)](
        src, lens, out, n, stride, block, num_warps=1
    )
    return out


__all__ = ["fill_bonus_tokens"]
