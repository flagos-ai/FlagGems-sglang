# SPDX-License-Identifier: Apache-2.0
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
"""Fixed TP8 MUSA launch configurations from the native SGLang image."""
from functools import cache

from .jit import load_musa_jit, load_musa_pybind


def _flags(shot):
    threads, blocks = (512, 120) if shot == 4 else (768, 80)
    return (
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        f"-DSGL_CUSTOM_AR_THREADS={threads}",
        f"-DSGL_CUSTOM_AR_BLOCKS={blocks}",
        "-DSGL_CUSTOM_AR_MAX_BLOCKS=120",
        "-DSGL_CUSTOM_AR_ATOMIC_BARRIER=1",
        "-DSGL_CUSTOM_AR_VECTOR_LOAD=0",
        f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={0 if shot == 4 else 1}",
    )


@cache
def module(shot):
    return load_musa_jit(
        f"flaggems_musa_ar_tp8_s{shot}",
        ("distributed/custom_all_reduce.mu",),
        extra_musa_cflags=_flags(shot),
    )


@cache
def pybind_module(shot):
    name = f"flaggems_musa_ar_tp8_s{shot}_pybind"
    return load_musa_pybind(
        name,
        (
            "distributed/custom_all_reduce.mu",
            "distributed/custom_all_reduce_pybind.cpp",
        ),
        extra_cflags=(f"-DSGL_CUSTOM_AR_TORCH_NS={name}",),
        extra_musa_cflags=_flags(shot),
    )


def prepare():
    for shot in (2, 4):
        module(shot)
        pybind_module(shot)
    return int(module(2).sgl_musa_custom_ar_meta_size())
