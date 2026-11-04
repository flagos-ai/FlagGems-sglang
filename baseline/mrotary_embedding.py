# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Six-case mRoPE baseline, original SGLang v0.5.11 Triton.

Set FLAGGEMS_SGLANG_SOURCE to the checkout containing the native reference and
workload builders. No installed SGLang fallback is allowed: its ABI/version may
have changed. This recipe never times the portable Torch reference.
"""

import hashlib
import importlib.util
import os
from pathlib import Path

import torch

OP_NAME = "mrotary_embedding"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
BENCH_RETURN_MODE = "median"  # Match SGLang's NVIDIA kernel timer.


def _path(name):
    root = os.environ.get("FLAGGEMS_SGLANG_SOURCE")
    if not root:
        raise RuntimeError(
            "set FLAGGEMS_SGLANG_SOURCE to the pinned FlagGems-sglang checkout"
        )
    return Path(root) / "src/flaggems_sglang/reference" / (name + ".py")


def _load(name):
    path = _path(name)
    spec = importlib.util.spec_from_file_location(
        "_baseline_" + name.replace("/", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native():
    return _load("native/mrotary_embedding").reference


def grid():
    return [dict(row) for row in _load("mrotary_embedding_cases").WORKLOADS]


def build_inputs(binding, dtype, device):
    return (), _load("mrotary_embedding_cases").build_inputs(binding, dtype, device)


def key_shape(binding):
    # Use meta tensors only to describe exact layout/ABI; never allocate CUDA here.
    cases = _load("mrotary_embedding_cases")
    return cases.baseline_key(cases.build_inputs(binding, DTYPES[0], "meta"))


def config(binding, dtype):
    cases = _load("mrotary_embedding_cases")
    values = cases.build_inputs(binding, dtype, "meta")
    inputs = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            inputs[name] = dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                dtype=str(value.dtype),
                storage_offset=value.storage_offset(),
            )
        else:
            inputs[name] = dict(scalar=value)
    return dict(
        inputs=inputs,
        outputs={name: inputs[name] for name in ("q", "k")},
        source=dict(
            upstream="sglang v0.5.11",
            upstream_blob="0a8dc2c33c7bbb2acb7e02b10796d210c5c079fc",
            native_sha256=hashlib.sha256(
                _path("native/mrotary_embedding").read_bytes()
            ).hexdigest(),
            cases_sha256=hashlib.sha256(
                _path("mrotary_embedding_cases").read_bytes()
            ).hexdigest(),
        ),
    )


def _reset_module():
    root = Path(os.environ["FLAGGEMS_SGLANG_SOURCE"])
    path = root / "src/flaggems_sglang/testing/reset_timing.py"
    spec = importlib.util.spec_from_file_location("_mrope_reset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input_reset(**inputs):
    return _reset_module().capture_reset(OP_NAME, inputs)


def time_with_reset(fn, reset):
    return _reset_module().measure_with_reset(fn, reset, torch.cuda)


def timing_contract():
    return dict(_reset_module().TIMING)
