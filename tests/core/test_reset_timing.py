# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""State-degeneration and excluded-reset timing regression controls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def module():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/flaggems_sglang/testing/reset_timing.py"
    )
    spec = importlib.util.spec_from_file_location("_reset_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_call_sees_original_and_reset_cache_cost_is_excluded(
    monkeypatch,
):
    mod = module()
    value = torch.tensor([2.0])
    reset = mod.capture_reset(
        "_fused_moe_kernel_sequence", dict(hidden_states=value)
    )
    clock = [0.0]
    calls = []
    original_reset = mod.InputReset.__call__

    def restore(plan):
        clock[0] += 50
        original_reset(plan)

    monkeypatch.setattr(mod.InputReset, "__call__", restore)
    monkeypatch.setattr(
        reset,
        "make_cache",
        lambda: SimpleNamespace(
            zero_=lambda: clock.__setitem__(0, clock[0] + 101)
        ),
    )

    class Event:
        def __init__(self, **kwargs):
            self.timestamp = None

        def record(self):
            self.timestamp = clock[0]

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    def op():
        calls.append(value.item())
        value.zero_()
        clock[0] += 7

    api = SimpleNamespace(Event=Event, synchronize=lambda: None)
    assert (
        mod.measure_with_reset(op, reset, api, warmup_ms=0, rep_ms=1000) == 7
    )
    assert len(calls) > 6 and set(calls) == {2.0}
    assert value.item() == 2.0


def test_pool_reset_uses_selected_views_not_advanced_index_copies():
    mod = module()
    pool = torch.arange(16.0).reshape(4, 1, 2, 2)
    original = pool.clone()
    reset = mod.capture_reset(
        "chunk_gated_delta_rule_fwd",
        dict(
            initial_state=pool, initial_state_indices=torch.tensor([-1, 1, 1])
        ),
    )
    assert len(reset.copies) == 1
    assert reset.copies[0][0].data_ptr() == pool[1].data_ptr()
    pool[1].zero_()
    pool[0].fill_(999)
    reset()
    assert torch.equal(pool[1], original[1])
    assert torch.equal(pool[0], torch.full_like(pool[0], 999))
    assert torch.equal(pool[2:], original[2:])


def test_reset_preserves_noncontiguous_layout_and_rejects_reallocation():
    mod = module()
    backing = torch.arange(40.0).reshape(4, 10)
    view = backing[:, 2:5]
    other = torch.ones_like(view)
    reset = mod.capture_reset(
        "gemma_fused_add_rmsnorm", dict(input=view, residual=other)
    )
    shape, stride, offset, pointer = (
        view.shape,
        view.stride(),
        view.storage_offset(),
        view.untyped_storage().data_ptr(),
    )
    original = view.clone()
    view.zero_()
    reset()
    assert torch.equal(view, original)
    assert (
        view.shape,
        view.stride(),
        view.storage_offset(),
        view.untyped_storage().data_ptr(),
    ) == (shape, stride, offset, pointer)
    view.resize_(1000)
    with pytest.raises(ValueError, match="geometry"):
        reset()


def test_readonly_workloads_do_not_opt_in_and_bad_slots_fail():
    mod = module()
    assert mod.reset_factory("gemma_rmsnorm") is None
    assert mod.capture_reset("silu_and_mul_out", {}) is None
    with pytest.raises(ValueError, match="outside pool"):
        mod.capture_reset(
            "chunk_gated_delta_rule_fwd",
            dict(
                initial_state=torch.zeros(2, 1, 1, 1),
                initial_state_indices=torch.tensor([2]),
            ),
        )


def test_event_failure_restores_original_input(monkeypatch):
    mod = module()
    value = torch.tensor([4.0])
    reset = mod.capture_reset(
        "_fused_moe_kernel_sequence", dict(hidden_states=value)
    )
    monkeypatch.setattr(
        reset, "make_cache", lambda: SimpleNamespace(zero_=lambda: None)
    )

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0

    with pytest.raises(RuntimeError, match="unavailable"):
        mod.measure_with_reset(
            lambda: value.zero_(),
            reset,
            SimpleNamespace(Event=Event, synchronize=lambda: None),
        )
    assert value.item() == 4


def test_reference_and_candidate_share_the_original_snapshot():
    # Execute the real phase orchestration without importing a GPU runtime.
    import ast
    import contextlib

    source = (
        Path(__file__).resolve().parents[2] / "benchmark/performance_utils.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_measure_input"
    )
    namespace = dict(
        BenchmarkMetrics=lambda **kw: SimpleNamespace(**kw),
        reference_execution=lambda: None,
        reference_mode_enabled=lambda: False,
        Config=SimpleNamespace(base_data=None),
        pytest=pytest,
        gc=SimpleNamespace(collect=lambda: None),
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    mod = module()
    value = torch.tensor([3.0])
    captures, observations = [], []
    bench = SimpleNamespace(
        to_bench_metrics=["latency_base", "latency", "speedup"],
        torch_op="reference",
        unpack_to_args_kwargs=lambda v: ((), dict(hidden_states=v)),
        record_shapes=lambda **kw: {},
        _candidate_call=lambda: ("candidate", contextlib.nullcontext(), None),
    )

    def capture(**kw):
        captures.append(value.item())
        return mod.capture_reset("_fused_moe_kernel_sequence", kw)

    bench.input_reset_factory = capture

    def latency(op, **kw):
        for _ in range(3):
            bench._active_input_reset()
            observations.append((op, value.item()))
            value.zero_()
        return 7

    bench.get_latency = latency
    result = namespace["_measure_input"](bench, value)
    assert captures == [3.0]
    assert observations == [("reference", 3.0)] * 3 + [("candidate", 3.0)] * 3
    assert result.speedup == 1
    assert bench._active_input_reset is None and value.item() == 3


def test_legacy_latency_keeps_the_original_timer():
    import ast

    source = (
        Path(__file__).resolve().parents[2] / "benchmark/performance_utils.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_latency"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    calls = []
    bench = SimpleNamespace(
        _benchmark_callable=lambda *a, **kw: ("fn", "xs"),
        _time_callable=lambda *a: calls.append(a) or 13,
    )
    assert namespace["get_latency"](bench, "op") == 13
    assert calls == [("fn", "xs")]


@pytest.mark.parametrize("enabled", [False, True])
def test_profile_repeated_captures_exclude_restoration(enabled):
    import ast
    import contextlib

    source = (
        Path(__file__).resolve().parents[2] / "benchmark/performance_utils.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_profile_cases"
    )
    events, seen = [], []
    config = SimpleNamespace(
        available_case_ids=set(),
        executed_case_ids=set(),
        profile_iterations=3,
        profile_warmup=2,
    )

    @contextlib.contextmanager
    def scope(**kwargs):
        events.append("start")
        yield
        events.append("end")

    config.profile_hook = scope
    namespace = dict(
        Optional=object,
        Collection=__import__("typing").Collection,
        Config=config,
        vendor_name="test",
        nullcontext=contextlib.nullcontext,
        torch_device_fn=SimpleNamespace(
            synchronize=lambda: events.append("sync")
        ),
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )

    def op(*, value):
        seen.append(value.item())
        events.append("op")
        value.zero_()

    class Reset:
        def __init__(self, value):
            self.value = value

        def make_cache(self):
            return object()

        def __call__(self):
            self.value.fill_(3)

        def prepare(self, cache):
            events.append("reset/cache")
            self()

    bench = SimpleNamespace(
        supports_cases=lambda: True,
        op_name="probe",
        _collect_cases=lambda: [SimpleNamespace(case_id="case")],
        build_inputs=lambda c: torch.tensor([3.0]),
        unpack_to_args_kwargs=lambda v: ((), dict(value=v)),
        _candidate_call=lambda: (op, contextlib.nullcontext(), False),
        input_reset_factory=(lambda **kw: Reset(**kw)) if enabled else None,
    )
    assert namespace["_run_profile_cases"](bench, ["case"]) == ["case"]
    assert seen == ([3] * 5 if enabled else [3, 0, 3, 0, 0])
    assert events.count("start") == (3 if enabled else 1)
    active = False
    for event in events:
        if event == "start":
            active = True
        elif event == "end":
            active = False
        elif event == "reset/cache":
            assert not active
