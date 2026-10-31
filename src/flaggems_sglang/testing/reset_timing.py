# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared, opt-in original-input timing; no runtime/backend imports."""

import json
import math
import statistics
from pathlib import Path

import torch

SPEC = json.loads(Path(__file__).with_suffix(".json").read_text())
TIMING = SPEC["timing"]


def _geometry(value):
    storage = value.untyped_storage()
    return (
        value.shape,
        value.stride(),
        value.dtype,
        value.device,
        value.storage_offset(),
        storage.data_ptr(),
        storage.nbytes(),
    )


class InputReset:
    def __init__(self, inputs, names):
        self.guards = []
        self.copies = []
        for name, indices_name in names.items():
            value = inputs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Read-write input {name} must be a tensor")
            self.guards.append(
                (value, value.untyped_storage(), _geometry(value))
            )
            if indices_name is None:
                views = [value]
            else:
                indices = inputs[indices_name]
                if indices.dtype not in (torch.int32, torch.int64):
                    raise TypeError("State slot indices must be int32/int64")
                slots = sorted(
                    {
                        int(x)
                        for x in indices.detach().cpu().reshape(-1).tolist()
                        if x >= 0
                    }
                )
                if any(slot >= value.shape[0] for slot in slots):
                    raise ValueError("State slot outside pool")
                # Integer indexing returns views; advanced indexing would copy.
                views = [value[slot] for slot in slots]
            self.copies.extend((view, view.detach().clone()) for view in views)
        if not self.guards:
            raise ValueError("No read-write inputs")
        self.device = self.guards[0][0].device
        if any(value.device != self.device for value, _, _ in self.guards):
            raise ValueError("Read-write inputs must share one device")

    def __call__(self):
        for value, _, geometry in self.guards:
            if _geometry(value) != geometry:
                raise ValueError(
                    "Read-write input geometry/allocation changed"
                )
        with torch.no_grad():
            for destination, original in self.copies:
                destination.copy_(original)

    def make_cache(self):
        return torch.empty(
            TIMING["cache_bytes"], dtype=torch.uint8, device=self.device
        )

    def prepare(self, cache):
        self()
        cache.zero_()


def capture_reset(operator, inputs):
    names = SPEC["read_write"].get(operator)
    return InputReset(inputs, names) if names else None


def reset_factory(operator):
    if operator not in SPEC["read_write"]:
        return None

    def capture(*args, **inputs):
        if args:
            raise TypeError("Reset-enabled workload requires keyword inputs")
        return capture_reset(operator, inputs)

    return capture


def measure_with_reset(fn, reset, device_api, *, warmup_ms=None, rep_ms=None):
    """Milliseconds, fixed cold-cache policy, reset/cache work outside samples.

    Calibration includes excluded preparation ONLY to budget iteration counts.
    Distinct events bracket just the operator in every published sample.
    """
    warmup_ms = TIMING["warmup_ms"] if warmup_ms is None else warmup_ms
    rep_ms = TIMING["rep_ms"] if rep_ms is None else rep_ms
    if (
        not all(math.isfinite(x) for x in (warmup_ms, rep_ms))
        or warmup_ms < 0
        or rep_ms <= 0
    ):
        raise ValueError("Invalid timing duration")
    cache = reset.make_cache()
    try:
        # First call, calibration, warmup and samples all start from originals.
        reset.prepare(cache)
        fn()
        device_api.synchronize()
        start, end = (device_api.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(5):
            reset.prepare(cache)
            fn()
        end.record()
        device_api.synchronize()
        estimate = start.elapsed_time(end) / 5
        if not math.isfinite(estimate) or estimate <= 0:
            raise RuntimeError(
                "Device event timing is unavailable or nonpositive"
            )
        warmups = min(TIMING["max_samples"], max(1, int(warmup_ms / estimate)))
        repeats = min(TIMING["max_samples"], max(1, int(rep_ms / estimate)))
        events = [
            (
                device_api.Event(enable_timing=True),
                device_api.Event(enable_timing=True),
            )
            for _ in range(repeats)
        ]
        for _ in range(warmups):
            reset.prepare(cache)
            fn()
        for begin, finish in events:
            reset.prepare(cache)
            begin.record()
            fn()
            finish.record()
        device_api.synchronize()
        times = [begin.elapsed_time(finish) for begin, finish in events]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise RuntimeError(
                "Device events produced an invalid operator latency"
            )
        return statistics.median(times)
    finally:
        # Also keep torch/candidate timing phases and later cases independent.
        reset()
        device_api.synchronize()
