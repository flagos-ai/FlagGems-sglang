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

SOURCE = (
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "union bits210 { float f; int i; };\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef int i32v __attribute__((ext_vector_type(16)));\n"
    "union vbits210 { f32v f; i32v i; };\n"
    "__attribute__((shared)) static int counts210[64];\n"
    "__attribute__((shared)) static int lower210[64];\n"
    "__attribute__((shared)) static int upper210[64];\n"
    "__attribute__((shared)) static float sums210[64];\n"
    "__attribute__((shared)) static int results210[4];\n"
    "__attribute__((shared)) static int tree_counts210[8];\n"
    "__attribute__((shared)) static int tree_lower210[8];\n"
    "__attribute__((shared)) static int tree_upper210[8];\n"
    "__attribute__((shared)) static float tree_sums210[8];\n"
    "__attribute__((shared)) static float selected_sum210;\n"
    "\n"
    'extern "C" __attribute__((device)) void _sparse210(gf32* out, con'
    "st gf32* probs, long long row, int vocab, int k) {\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    int chunk = (((vocab + 63) / 64) + 15) & -16;\n"
    "    int start = core * chunk;\n"
    "    int length = vocab - start;\n"
    "    if (length > chunk) length = chunk;\n"
    "    if (length < 0) length = 0;\n"
    "    float storage[1792] __attribute__((aligned(64)));\n"
    "    float* buffer = storage;\n"
    "    float* values = storage + 1536;\n"
    "    int* positions = (int*)(storage + 1664);\n"
    "    if (core == 0) {\n"
    "        int size = vocab < 512 ? vocab : 512;\n"
    "        __builtin_xpu_gm2lm_v3(buffer, probs + (long long)row * v"
    "ocab, 0, size * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        float maximum = 0.0f;\n"
    "        i32v peak = (i32v)(0);\n"
    "        int j = 0;\n"
    "        for (; j + 15 < size; j += 16) {\n"
    "            vbits210 b; b.f = *((f32v*)(buffer + j));\n"
    "            i32v bits = b.i & (i32v)(2147483647);\n"
    "            i32v mask = (peak - bits) >> 31;\n"
    "            peak ^= (bits ^ peak) & mask;\n"
    "        }\n"
    "        { bits210 p; p.i = peak[0]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[1]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[2]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[3]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[4]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[5]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[6]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[7]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[8]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[9]; if (p.f > maximum) maximum = "
    "p.f; }\n"
    "        { bits210 p; p.i = peak[10]; if (p.f > maximum) maximum ="
    " p.f; }\n"
    "        { bits210 p; p.i = peak[11]; if (p.f > maximum) maximum ="
    " p.f; }\n"
    "        { bits210 p; p.i = peak[12]; if (p.f > maximum) maximum ="
    " p.f; }\n"
    "        { bits210 p; p.i = peak[13]; if (p.f > maximum) maximum ="
    " p.f; }\n"
    "        { bits210 p; p.i = peak[14]; if (p.f > maximum) maximum ="
    " p.f; }\n"
    "        { bits210 p; p.i = peak[15]; if (p.f > maximum) maximum ="
    " p.f; }\n"
    "        for (; j < size; j++) if (buffer[j] > maximum) maximum = "
    "buffer[j];\n"
    "        sums210[0] = maximum;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    float sample = sums210[0];\n"
    "    float factor = 1.0f - 1.5f * (float)k / (float)vocab;\n"
    "    if (factor < 0.0f) factor = 0.0f;\n"
    "    bits210 cutoff; cutoff.f = sample * factor;\n"
    "    int local_count = 0;\n"
    "    int lo = 0, hi = 0;\n"
    "    if (length > 0) {\n"
    "        int first_size = length < 768 ? length : 768;\n"
    "        __builtin_xpu_gm2lm_v3(storage, probs + (long long)row * "
    "vocab + start, 0, first_size * 4);\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int offset = 0; offset < length; offset += 768) {\n"
    "        buffer = storage + ((offset / 768) & 1) * 768;\n"
    "        int size = length - offset;\n"
    "        if (size > 768) size = 768;\n"
    "        if (offset + 768 < length) {\n"
    "            int next_size = length - offset - 768;\n"
    "            if (next_size > 768) next_size = 768;\n"
    "            float* next_buffer = storage + (1 - ((offset / 768) &"
    " 1)) * 768;\n"
    "            __builtin_xpu_gm2lm_v3(next_buffer, probs + (long lon"
    "g)row * vocab + start + offset + 768, 0, next_size * 4);\n"
    "        }\n"
    "        int j = 0;\n"
    "        for (; j + 15 < size; j += 16) {\n"
    "            float prefetched0 = buffer[j + 0];\n"
    "            float prefetched1 = buffer[j + 1];\n"
    "            float prefetched2 = buffer[j + 2];\n"
    "            float prefetched3 = buffer[j + 3];\n"
    "            float prefetched4 = buffer[j + 4];\n"
    "            float prefetched5 = buffer[j + 5];\n"
    "            float prefetched6 = buffer[j + 6];\n"
    "            float prefetched7 = buffer[j + 7];\n"
    "            float prefetched8 = buffer[j + 8];\n"
    "            float prefetched9 = buffer[j + 9];\n"
    "            float prefetched10 = buffer[j + 10];\n"
    "            float prefetched11 = buffer[j + 11];\n"
    "            float prefetched12 = buffer[j + 12];\n"
    "            float prefetched13 = buffer[j + 13];\n"
    "            float prefetched14 = buffer[j + 14];\n"
    "            float prefetched15 = buffer[j + 15];\n"
    "            {\n"
    "            bits210 b; b.f = prefetched0;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 0);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched1;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 1);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched2;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 2);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched3;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 3);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched4;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 4);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched5;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 5);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched6;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 6);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched7;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 7);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched8;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 8);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched9;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 9);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched10;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 10);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched11;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 11);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched12;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 12);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched13;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 13);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched14;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 14);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            {\n"
    "            bits210 b; b.f = prefetched15;\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + (j "
    "+ 15);\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            }\n"
    "            *((f32v*)(buffer + j)) = (f32v)(0.0f);\n"
    "        }\n"
    "        for (; j < size; j++) {\n"
    "            bits210 b; b.f = buffer[j];\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (b.f >= cutoff.f) {\n"
    "                if (bits > hi) hi = bits;\n"
    "                if (local_count < 128) {\n"
    "                    values[local_count] = b.f;\n"
    "                    positions[local_count] = start + offset + j;\n"
    "                }\n"
    "                local_count++;\n"
    "            }\n"
    "            buffer[j] = 0.0f;\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(out + (long long)row * vocab + sta"
    "rt + offset, buffer, 0, size * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    buffer = storage;\n"
    "    lower210[core] = lo;\n"
    "    upper210[core] = hi;\n"
    "    counts210[core] = local_count;\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    int candidate_count = 0, overflow = 0;\n"
    "    if (core < 8) {\n"
    "        int local_total = 0, local_overflow = 0, local_hi = 0;\n"
    "        for (int j = core * 8; j < core * 8 + 8; j++) {\n"
    "            int value = counts210[j];\n"
    "            local_total += value;\n"
    "            if (value > 128) local_overflow = 1;\n"
    "            if (upper210[j] > local_hi) local_hi = upper210[j];\n"
    "        }\n"
    "        tree_counts210[core] = local_total;\n"
    "        tree_lower210[core] = local_overflow;\n"
    "        tree_upper210[core] = local_hi;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    if (core == 0) {\n"
    "        for (int j = 0; j < 8; j++) {\n"
    "            candidate_count += tree_counts210[j];\n"
    "            overflow |= tree_lower210[j];\n"
    "            if (tree_upper210[j] > hi) hi = tree_upper210[j];\n"
    "        }\n"
    "        results210[0] = candidate_count;\n"
    "        results210[1] = overflow;\n"
    "        results210[2] = hi;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    candidate_count = results210[0];\n"
    "    overflow = results210[1];\n"
    "    hi = results210[2];\n"
    "    int compact = candidate_count >= k && overflow == 0;\n"
    "    if (compact) lo = cutoff.i & 2147483647;\n"
    "    if (k == 1) lo = hi;\n"
    "    if (k == vocab) hi = lo;\n"
    "    int nlo = compact ? candidate_count : vocab, nhi = 0, step = "
    "0;\n"
    "    int have_total = 0;\n"
    "    while (lo < hi) {\n"
    "        bits210 lf, hf, estimate;\n"
    "        lf.i = lo; hf.i = hi;\n"
    "        estimate.f = lf.f + (hf.f - lf.f) * ((float)(nlo-k) / (fl"
    "oat)(nlo-nhi > 0 ? nlo-nhi : 1));\n"
    "        int mid = estimate.i;\n"
    "        if (mid < lo + 1) mid = lo + 1;\n"
    "        if (mid > hi) mid = hi;\n"
    "        if (step % 4 == 3) mid = lo + (hi-lo+1) / 2;\n"
    "        int count = 0, lower = 0, upper = 2147483647;\n"
    "        float iteration_sum = 0.0f;\n"
    "        int active_length = compact ? local_count : length;\n"
    "        for (int offset = 0; offset < active_length; offset += 10"
    "24) {\n"
    "            int size = active_length - offset;\n"
    "            if (size > 1024) size = 1024;\n"
    "            if (!compact) {\n"
    "                __builtin_xpu_gm2lm_v3(buffer, probs + (long long"
    ")row * vocab + start + offset, 0, size * 4);\n"
    "                __builtin_xpu2_mfence(7);\n"
    "            }\n"
    "            float* active = compact ? values : buffer;\n"
    "            for (int j = 0; j < size; j++) {\n"
    "                bits210 b; b.f = active[j];\n"
    "                int bits = b.i & 2147483647;\n"
    "                if (bits >= mid) { count++; iteration_sum += b.f;"
    " if (bits < upper) upper = bits; }\n"
    "                else if (bits > lower) lower = bits;\n"
    "            }\n"
    "        }\n"
    "        counts210[core] = count;\n"
    "        sums210[core] = iteration_sum;\n"
    "        lower210[core] = lower;\n"
    "        upper210[core] = upper;\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        if (core < 8) {\n"
    "            int group_count = 0, group_lower = 0, group_upper = 2"
    "147483647;\n"
    "            float group_sum = 0.0f;\n"
    "            for (int j = core * 8; j < core * 8 + 8; j++) {\n"
    "                group_count += counts210[j];\n"
    "                group_sum += sums210[j];\n"
    "                if (lower210[j] > group_lower) group_lower = lowe"
    "r210[j];\n"
    "                if (upper210[j] < group_upper) group_upper = uppe"
    "r210[j];\n"
    "            }\n"
    "            tree_counts210[core] = group_count;\n"
    "            tree_sums210[core] = group_sum;\n"
    "            tree_lower210[core] = group_lower;\n"
    "            tree_upper210[core] = group_upper;\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        if (core == 0) {\n"
    "            count = 0; lower = 0; upper = 2147483647;\n"
    "            float global_sum = 0.0f;\n"
    "            for (int j = 0; j < 8; j++) {\n"
    "                count += tree_counts210[j];\n"
    "                global_sum += tree_sums210[j];\n"
    "                if (tree_lower210[j] > lower) lower = tree_lower2"
    "10[j];\n"
    "                if (tree_upper210[j] < upper) upper = tree_upper2"
    "10[j];\n"
    "            }\n"
    "            if (count >= k) selected_sum210 = global_sum;\n"
    "            results210[0] = count;\n"
    "            results210[1] = lower;\n"
    "            results210[2] = upper;\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        count = results210[0];\n"
    "        lower = results210[1];\n"
    "        upper = results210[2];\n"
    "        if (count >= k) { lo = upper; nlo = count; have_total = 1"
    "; }\n"
    "        if (count <= k) hi = count == k ? upper : lower;\n"
    "        if (count < k) nhi = count;\n"
    "        step++;\n"
    "    }\n"
    "    bits210 pivot; pivot.i = lo;\n"
    "    float total = 0.0f;\n"
    "    if (!have_total) {\n"
    "    int active_length = compact ? local_count : length;\n"
    "    for (int offset = 0; offset < active_length; offset += 1024) "
    "{\n"
    "        int size = active_length - offset;\n"
    "        if (size > 1024) size = 1024;\n"
    "        if (!compact) {\n"
    "            __builtin_xpu_gm2lm_v3(buffer, probs + (long long)row"
    " * vocab + start + offset, 0, size * 4);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "        float* active = compact ? values : buffer;\n"
    "        for (int j = 0; j < size; j++) if (active[j] >= pivot.f) "
    "total += active[j];\n"
    "    }\n"
    "    sums210[core] = total;\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    if (core < 8) {\n"
    "        float subtotal = 0.0f;\n"
    "        for (int j = core * 8; j < core * 8 + 8; j++) subtotal +="
    " sums210[j];\n"
    "        tree_sums210[core] = subtotal;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    if (core == 0) {\n"
    "        total = 0.0f;\n"
    "        for (int j = 0; j < 8; j++) total += tree_sums210[j];\n"
    "        sums210[0] = total;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    total = sums210[0];\n"
    "    } else { total = selected_sum210; }\n"
    "    float inv = 1.0f / total;\n"
    "    if (compact) {\n"
    "        for (int j = 0; j < local_count; j++) {\n"
    "            if (values[j] >= pivot.f) out[(long long)row * vocab "
    "+ positions[j]] = values[j] * inv;\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    } else {\n"
    "    for (int offset = 0; offset < length; offset += 1024) {\n"
    "        int size = length - offset;\n"
    "        if (size > 1024) size = 1024;\n"
    "        __builtin_xpu_gm2lm_v3(buffer, probs + (long long)row * v"
    "ocab + start + offset, 0, size * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int j = 0; j < size; j++) buffer[j] = buffer[j] >= p"
    "ivot.f ? buffer[j] * inv : 0.0f;\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(out + (long long)row * vocab + sta"
    "rt + offset, buffer, 0, size * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    }\n"
    "}\n"
    'extern "C" __attribute__((device)) void _cooperative210(gf32* out'
    ", const gf32* probs, long long row, int vocab, int k) {\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    int chunk = (vocab + 63) / 64;\n"
    "    int start = core * chunk;\n"
    "    int length = vocab - start;\n"
    "    if (length > chunk) length = chunk;\n"
    "    if (length < 0) length = 0;\n"
    "    float values[128] __attribute__((aligned(64)));\n"
    "    if (length > 0) {\n"
    "        __builtin_xpu_gm2lm_v3(values, probs + (long long)row * v"
    "ocab + start, 0, length * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    int lo = 0, hi = 0;\n"
    "    for (int j = 0; j < length; j++) {\n"
    "        bits210 b; b.f = values[j];\n"
    "        int bits = b.i & 2147483647;\n"
    "        if (bits > hi) hi = bits;\n"
    "    }\n"
    "    upper210[core] = hi;\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    if (core == 0) {\n"
    "        for (int j = 0; j < 64; j++) if (upper210[j] > hi) hi = u"
    "pper210[j];\n"
    "        results210[0] = hi;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    hi = results210[0];\n"
    "    if (k == 1) lo = hi;\n"
    "    if (k == vocab) hi = lo;\n"
    "    int nlo = vocab, nhi = 0, step = 0;\n"
    "    while (lo < hi) {\n"
    "        bits210 lf, hf, estimate;\n"
    "        lf.i = lo; hf.i = hi;\n"
    "        estimate.f = lf.f + (hf.f - lf.f) * ((float)(nlo-k) / (fl"
    "oat)(nlo-nhi > 0 ? nlo-nhi : 1));\n"
    "        int mid = estimate.i;\n"
    "        if (mid < lo + 1) mid = lo + 1;\n"
    "        if (mid > hi) mid = hi;\n"
    "        if (step % 4 == 3) mid = lo + (hi-lo+1) / 2;\n"
    "        int count = 0, lower = 0, upper = 2147483647;\n"
    "        for (int j = 0; j < length; j++) {\n"
    "            bits210 b; b.f = values[j];\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (bits >= mid) { count++; if (bits < upper) upper ="
    " bits; }\n"
    "            else if (bits > lower) lower = bits;\n"
    "        }\n"
    "        counts210[core] = count;\n"
    "        lower210[core] = lower;\n"
    "        upper210[core] = upper;\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        if (core == 0) {\n"
    "            count = 0; lower = 0; upper = 2147483647;\n"
    "            for (int j = 0; j < 64; j++) {\n"
    "                count += counts210[j];\n"
    "                if (lower210[j] > lower) lower = lower210[j];\n"
    "                if (upper210[j] < upper) upper = upper210[j];\n"
    "            }\n"
    "            results210[0] = count;\n"
    "            results210[1] = lower;\n"
    "            results210[2] = upper;\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        count = results210[0];\n"
    "        lower = results210[1];\n"
    "        upper = results210[2];\n"
    "        if (count >= k) { lo = upper; nlo = count; }\n"
    "        if (count <= k) hi = count == k ? upper : lower;\n"
    "        if (count < k) nhi = count;\n"
    "        step++;\n"
    "    }\n"
    "    bits210 pivot; pivot.i = lo;\n"
    "    float total = 0.0f;\n"
    "    for (int j = 0; j < length; j++) if (values[j] >= pivot.f) to"
    "tal += values[j];\n"
    "    sums210[core] = total;\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    if (core == 0) {\n"
    "        total = 0.0f;\n"
    "        for (int j = 0; j < 64; j++) total += sums210[j];\n"
    "        sums210[0] = total;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    total = sums210[0];\n"
    "    float inv = 1.0f / total;\n"
    "    for (int j = 0; j < length; j++) values[j] = values[j] >= piv"
    "ot.f ? values[j] * inv : 0.0f;\n"
    "    if (length > 0) {\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(out + (long long)row * vocab + sta"
    "rt, values, 0, length * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
    "__attribute__((shared)) static long long widecounts210[64];\n"
    "__attribute__((shared)) static int widelower210[64];\n"
    "__attribute__((shared)) static int wideupper210[64];\n"
    "__attribute__((shared)) static float widesums210[64];\n"
    "\n"
    'extern "C" __attribute__((device)) void _wide210(gf32* out, const'
    " gf32* probs, long long row, long long vocab, long long k) {\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    long long chunk = (vocab + 63) / 64;\n"
    "    long long start = core * chunk;\n"
    "    long long length = vocab - start;\n"
    "    if (length > chunk) length = chunk;\n"
    "    if (length < 0) length = 0;\n"
    "    float values[1024] __attribute__((aligned(64)));\n"
    "    int lo = 2147483647, hi = 0;\n"
    "    for (long long offset = 0; offset < length; offset += 1024) {"
    "\n"
    "        long long size = length - offset;\n"
    "        if (size > 1024) size = 1024;\n"
    "        __builtin_xpu_gm2lm_v3(values, probs + (long long)row * v"
    "ocab + start + offset, 0, size * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int j = 0; j < size; j++) {\n"
    "            bits210 b; b.f = values[j];\n"
    "            int bits = b.i & 2147483647;\n"
    "            if (bits < lo) lo = bits;\n"
    "            if (bits > hi) hi = bits;\n"
    "        }\n"
    "    }\n"
    "    widelower210[core] = lo;\n"
    "    wideupper210[core] = hi;\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    for (int j = 0; j < 64; j++) {\n"
    "        if (widelower210[j] < lo) lo = widelower210[j];\n"
    "        if (wideupper210[j] > hi) hi = wideupper210[j];\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    if (k == 1) lo = hi;\n"
    "    if (k == vocab) hi = lo;\n"
    "    long long nlo = vocab, nhi = 0; int step = 0;\n"
    "    while (lo < hi) {\n"
    "        bits210 lf, hf, estimate;\n"
    "        lf.i = lo; hf.i = hi;\n"
    "        estimate.f = lf.f + (hf.f - lf.f) * ((float)(nlo-k) / (fl"
    "oat)(nlo-nhi > 0 ? nlo-nhi : 1));\n"
    "        int mid = estimate.i;\n"
    "        if (mid < lo + 1) mid = lo + 1;\n"
    "        if (mid > hi) mid = hi;\n"
    "        if (step % 4 == 3) mid = lo + (hi-lo+1) / 2;\n"
    "        long long count = 0; int lower = 0, upper = 2147483647;\n"
    "        for (long long offset = 0; offset < length; offset += 102"
    "4) {\n"
    "            long long size = length - offset;\n"
    "            if (size > 1024) size = 1024;\n"
    "            if (length > 1024) {\n"
    "                __builtin_xpu_gm2lm_v3(values, probs + (long long"
    ")row * vocab + start + offset, 0, size * 4);\n"
    "                __builtin_xpu2_mfence(7);\n"
    "            }\n"
    "            for (int j = 0; j < size; j++) {\n"
    "                bits210 b; b.f = values[j];\n"
    "                int bits = b.i & 2147483647;\n"
    "                if (bits >= mid) { count++; if (bits < upper) upp"
    "er = bits; }\n"
    "                else if (bits > lower) lower = bits;\n"
    "            }\n"
    "        }\n"
    "        widecounts210[core] = count;\n"
    "        widelower210[core] = lower;\n"
    "        wideupper210[core] = upper;\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        count = 0; lower = 0; upper = 2147483647;\n"
    "        for (int j = 0; j < 64; j++) {\n"
    "            count += widecounts210[j];\n"
    "            if (widelower210[j] > lower) lower = widelower210[j];"
    "\n"
    "            if (wideupper210[j] < upper) upper = wideupper210[j];"
    "\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_csr_set_sync_group(65535);\n"
    "        if (count >= k) { lo = upper; nlo = count; }\n"
    "        if (count <= k) hi = count == k ? upper : lower;\n"
    "        if (count < k) nhi = count;\n"
    "        step++;\n"
    "    }\n"
    "    bits210 pivot; pivot.i = lo;\n"
    "    float total = 0.0f, correction = 0.0f;\n"
    "    for (long long offset = 0; offset < length; offset += 1024) {"
    "\n"
    "        long long size = length - offset;\n"
    "        if (size > 1024) size = 1024;\n"
    "        if (length > 1024) {\n"
    "            __builtin_xpu_gm2lm_v3(values, probs + (long long)row"
    " * vocab + start + offset, 0, size * 4);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "        for (int j = 0; j < size; j++) if (values[j] >= pivot.f) "
    "{\n"
    "            float value = values[j] - correction;\n"
    "            float updated = total + value;\n"
    "            correction = (updated - total) - value;\n"
    "            total = updated;\n"
    "        }\n"
    "    }\n"
    "    widesums210[core] = total;\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    total = 0.0f;\n"
    "    for (int j = 0; j < 64; j++) total += widesums210[j];\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_csr_set_sync_group(65535);\n"
    "    float inv = 1.0f / total;\n"
    "    for (long long offset = 0; offset < length; offset += 1024) {"
    "\n"
    "        long long size = length - offset;\n"
    "        if (size > 1024) size = 1024;\n"
    "        if (length > 1024) {\n"
    "            __builtin_xpu_gm2lm_v3(values, probs + (long long)row"
    " * vocab + start + offset, 0, size * 4);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "        for (int j = 0; j < size; j++) values[j] = values[j] >= p"
    "ivot.f ? values[j] * inv : 0.0f;\n"
    "        __builtin_xpu_lm2gm_v3(out + (long long)row * vocab + sta"
    "rt + offset, values, 0, size * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect(
    "xpu3", source=SOURCE, flags=("-fno-fast-math", "-ffp-contract=off")
)
def _wide210(out, probs, row, vocab, k):
    pass


@tle.raw.dialect("xpu3", source=SOURCE, flags=("-ffp-contract=off",))
def _cooperative210(out, probs, row, vocab, k):
    pass


@tle.raw.dialect("xpu3", source=SOURCE, flags=("-ffp-contract=off",))
def _sparse210(out, probs, row, vocab, k):
    pass


@triton.jit
def _raw(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if V > 1048576:
        if KT:
            k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int64)
        else:
            k = tl.full((), KV, tl.int64)
        tle.raw.call(
            _wide210,
            outputs=(O,),
            inputs=(P, row, tl.full((), V, tl.int64), k),
        )
    else:
        if KT:
            k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
        else:
            k = tl.full((), KV, tl.int32)
        if V > 8192:
            tle.raw.call(
                _sparse210,
                outputs=(O,),
                inputs=(P, row, tl.full((), V, tl.int32), k),
            )
        else:
            tle.raw.call(
                _cooperative210,
                outputs=(O,),
                inputs=(P, row, tl.full((), V, tl.int32), k),
            )


@triton.jit
def _fixed(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, W)
    x = tl.load(P + row * V + tl.minimum(c, V - 1))
    x = tl.where(c < V, x, 0.0)
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    if not KT and KV == 1:
        pivot = tl.max(x, 0)
        kept = tl.where(x >= pivot, x, 0.0)
    elif not KT and KV == V:
        kept = x
    else:
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), 1065353216, tl.int32)
        for step in range(13):
            distance = hi - lo
            piece = tl.maximum(1, (distance + 4) // 5)
            m0 = tl.minimum(hi, lo + piece)
            m1 = tl.minimum(hi, lo + 2 * piece)
            m2 = tl.minimum(hi, lo + 3 * piece)
            m3 = tl.minimum(hi, lo + 4 * piece)
            packed = tl.sum(
                ((c < V) & (bits >= m0)).to(tl.uint32)
                + (((c < V) & (bits >= m1)).to(tl.uint32) << 8)
                + (((c < V) & (bits >= m2)).to(tl.uint32) << 16)
                + (((c < V) & (bits >= m3)).to(tl.uint32) << 24),
                0,
            ).to(tl.uint32)
            count0 = (packed & 255).to(tl.int32)
            count1 = (packed >> 8 & 255).to(tl.int32)
            count2 = (packed >> 16 & 255).to(tl.int32)
            count3 = (packed >> 24 & 255).to(tl.int32)
            slot = (
                (count0 >= k).to(tl.int32)
                + (count1 >= k).to(tl.int32)
                + (count2 >= k).to(tl.int32)
                + (count3 >= k).to(tl.int32)
            )
            next_lo = tl.minimum(hi, lo + slot * piece)
            next_hi = tl.minimum(hi, lo + (slot + 1) * piece - 1)
            lo = next_lo
            hi = next_hi
        kept = tl.where(bits >= lo, x, 0.0)
    total = tl.sum(kept, 0)
    tl.store(O + row * V + c, kept / total, c < V)


@triton.jit
def _copy_complete(P, Out, N: tl.constexpr, W: tl.constexpr):
    c = tl.program_id(0).to(tl.int64) * W + tl.arange(0, W)
    x = tl.load(P + tl.minimum(c, N - 1))
    tl.store(Out + c, x, c < N)


MAX_BATCH = 65535


def top_k_renorm(probs, top_k):
    n, v = probs.shape
    out = torch.empty_like(probs)
    if n > MAX_BATCH:
        for start in range(0, n, MAX_BATCH):
            end = min(start + MAX_BATCH, n)
            current_k = (
                top_k[start:end]
                if isinstance(top_k, torch.Tensor) and top_k.numel() > 1
                else top_k
            )
            current = top_k_renorm(probs[start:end], current_k)
            _copy_complete[triton.cdiv(current.numel(), 4096),](
                current,
                out[start:end],
                current.numel(),
                4096,
                num_warps=4,
                num_stages=1,
            )
        return out
    kt = isinstance(top_k, torch.Tensor)
    keys = top_k if kt else probs
    single = keys.numel() == 1 if kt else True
    ks = keys.stride(0) if kt else 0
    kv = 0 if kt else int(top_k)
    if v <= 128 or (v <= 8192 and (not kt) and (kv == v)):
        _fixed[n,](
            probs,
            keys,
            out,
            v,
            ks,
            kt,
            single,
            kv,
            max(2 if kv == v else 128, triton.next_power_of_2(v)),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return out
    _raw[n,](
        probs,
        keys,
        out,
        v,
        ks,
        kt,
        single,
        kv,
        num_warps=4,
        num_stages=1,
        isCloseUnrollControl=True,
    )
    return out


__all__ = ["top_k_renorm"]
