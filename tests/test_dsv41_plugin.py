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

"""Adapter lifecycle and scheduler device ordering, with no GPU execution."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def fresh():
    path = Path(__file__).parents[1] / "src/flaggems_sglang_plugin/__init__.py"
    spec = importlib.util.spec_from_file_location(
        "isolated_dsv41_plugin", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_without_scheduler_does_not_import_gpu_or_sglang(
    monkeypatch,
):
    p = fresh()
    before = set(sys.modules)
    p.load_plugin()
    assert set(sys.modules) == before
    assert p._configure_original is None


def test_device_is_set_before_operator_install(monkeypatch):
    p = fresh()
    events = []
    scheduler = types.ModuleType("sglang.srt.managers.scheduler")
    scheduler.configure_scheduler_process = (
        lambda *a, **k: events.append("configure") or 3
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda: (9, 0),
        set_device=lambda device: events.append(("device", device)),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers.scheduler", scheduler
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(p, "install", lambda: events.append("install"))
    p.load_plugin()
    assert scheduler.configure_scheduler_process(None, 5) == 3
    assert events == ["configure", ("device", 5), "install"]
    wrapped = scheduler.configure_scheduler_process
    p.load_plugin()
    assert scheduler.configure_scheduler_process is wrapped


def test_uninstall_preserves_subsequent_owner(monkeypatch):
    p = fresh()
    module = types.SimpleNamespace()
    old, new, other = (lambda: 1), (lambda: 2), (lambda: 3)
    module.a = new
    module.b = other
    handle = p.Installation(
        [(module, "a", old, new), (module, "b", old, new)], {}
    )
    p._handle = handle
    handle.uninstall()
    assert module.a is old and module.b is other and p._handle is None
    handle.uninstall()


@pytest.mark.parametrize(
    "available, capability", [(False, None), (True, (8, 0))]
)
def test_other_platforms_keep_native_scheduler(
    monkeypatch, available, capability
):
    p = fresh()
    scheduler = types.ModuleType("sglang.srt.managers.scheduler")
    scheduler.configure_scheduler_process = lambda *a, **k: "native"
    fake_torch = types.ModuleType("torch")
    devices = []
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        set_device=devices.append,
        get_device_capability=lambda: capability,
    )
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def unexpected():
        raise AssertionError("SM90 adapter installed on another platform")

    monkeypatch.setattr(p, "install", unexpected)
    p.load_plugin()
    assert scheduler.configure_scheduler_process(None, 2) == "native"
    assert devices == ([2] if available else [])


def test_unknown_source_fails_before_any_rebind(monkeypatch, tmp_path):
    p = fresh()
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        get_device_capability=lambda: (9, 0)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source = tmp_path / "op.py"
    source.write_text("unreviewed source")
    sentinel = lambda: None
    module = types.SimpleNamespace(__file__=str(source), sentinel=sentinel)
    monkeypatch.setattr(p.importlib, "import_module", lambda _: module)
    with pytest.raises(RuntimeError, match="unreviewed SGLang"):
        p.install()
    assert module.sentinel is sentinel and p._handle is None


def mock_reviewed_install(monkeypatch, tmp_path, plugin, *, abi=1):
    import hashlib

    def native(name):
        return lambda *args, **kwargs: ("native", name)

    modules = {}
    groups = [
        [
            "fp4_index_logits_decode",
            "fp4_index_logits_req_to_token",
            "fp4_index_logits_candidate_blocks",
            "quantize_bf16_index_queries_fp8",
            "unpack_fp4_index_keys_to_fp8",
            "fp8_index_logits_prefill",
        ],
        ["w8a8_block_fp8_matmul_triton"],
        [
            "hc_mix_stats",
            "hc_mix_stats_sinkhorn_exact",
            "hc_combine",
            "hc_mix_stats_sinkhorn",
        ],
    ]
    pins = {}
    originals = {}
    for i, (module_name, functions) in enumerate(
        zip(plugin._SOURCE_PINS, groups)
    ):
        file = tmp_path / f"op{i}.py"
        file.write_text(f"reviewed native source {i}")
        module = types.ModuleType(module_name)
        module.__file__ = str(file)
        for name in functions:
            fn = native(name)
            originals[name] = fn
            setattr(module, name, fn)
        pins[module_name] = hashlib.sha256(file.read_bytes()).hexdigest()
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(plugin, "_install_w4a8", lambda: None)
    monkeypatch.setattr(plugin, "_SOURCE_PINS", pins)
    monkeypatch.setattr(
        plugin.importlib, "import_module", lambda name: modules[name]
    )
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        get_device_capability=lambda: (9, 0), current_device=lambda: 0
    )
    torch._dynamo = types.SimpleNamespace(is_compiling=lambda: False)
    for name in ["float8_e4m3fn", "float32", "bfloat16", "float16"]:
        setattr(torch, name, object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    replacements = {}
    gems = types.ModuleType("flaggems_sglang")
    for name in groups[0][1:3]:
        fn = lambda *args, _name=name, **kwargs: ("framework", _name)
        setattr(gems, name, fn)
        replacements[name] = fn
    monkeypatch.setitem(sys.modules, "flaggems_sglang", gems)
    packages = {}
    for name in [
        "flag_gems",
        "flag_gems.ops",
        "flag_gems.fused",
        "flag_gems.fused.dsv41",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        packages[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    config = types.ModuleType("flag_gems.ops._w8a8_block_fp8_configs")
    config.BLOCK_FP8_SM90_ABI = abi
    config._H100_MIDRANGE_FP8_CONFIGS = {(6, 1792, 5120): None}
    core = types.ModuleType("flag_gems.fused.dsv41.fp4_indexer")
    core.FP4_INDEX_SCORE_ABI = 1
    core.fp4_index_logits_decode = lambda *a, **k: "core fp4"
    replacements["fp4_index_logits_decode"] = core.fp4_index_logits_decode
    mhc = types.ModuleType("flag_gems.fused.dsv41.mhc")
    mhc.MHC_EXACT_ABI = 1
    for name in groups[2]:
        fn = lambda *a, _name=name, **k: ("common", _name)
        setattr(mhc, name, fn)
        replacements[name] = fn
    matmul = types.ModuleType("flag_gems.ops.w8a8_block_fp8_matmul")
    calls = []
    matmul.w8a8_block_fp8_matmul = (
        lambda *a, **k: calls.append(a) or "common fp8"
    )
    for module in [config, core, mhc, matmul]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    packages["flag_gems.fused.dsv41"].mhc = mhc
    casts = types.ModuleType("isolated_dsv41_plugin.compat.dsv41.fp4_prefill")
    for name in groups[0][3:5]:
        fn = native("optimized " + name)
        setattr(casts, name, fn)
        replacements[name] = fn
    release_mhc = types.ModuleType("isolated_dsv41_plugin.compat.dsv41.mhc")
    release_mhc.hc_mix_stats_sinkhorn = native("optimized sinkhorn")
    replacements["hc_mix_stats_sinkhorn"] = release_mhc.hc_mix_stats_sinkhorn
    prefill = types.ModuleType("isolated_dsv41_plugin.prefill_compat")
    prefill.fp8_index_logits_prefill = native("optimized prefill")
    replacements["fp8_index_logits_prefill"] = prefill.fp8_index_logits_prefill
    for name in [
        "isolated_dsv41_plugin",
        "isolated_dsv41_plugin.compat",
        "isolated_dsv41_plugin.compat.dsv41",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for module in [casts, release_mhc, prefill]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    alias = types.ModuleType("sglang.test_aliases")
    for name, fn in originals.items():
        setattr(alias, name, fn)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    return alias, originals, replacements, calls, torch


def test_install_routes_common_math_and_framework_fusion(
    monkeypatch, tmp_path
):
    p = fresh()
    alias, originals, replacements, calls, torch = mock_reviewed_install(
        monkeypatch, tmp_path, p
    )
    p._SOURCE_PINS = {
        name: ("previous-source", digest)
        for name, digest in p._SOURCE_PINS.items()
    }
    handle = p.install()
    assert p.install() is handle
    for name, fn in replacements.items():
        assert getattr(alias, name) is fn

    class Tensor:
        ndim = 2

        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

        def is_contiguous(self):
            return True

    a, b = Tensor((6, 5120), torch.float8_e4m3fn), Tensor(
        (1792, 5120), torch.float8_e4m3fn
    )
    sa, sb = Tensor((6, 160), torch.float32), Tensor((56, 160), torch.float32)
    assert (
        alias.w8a8_block_fp8_matmul_triton(
            a, b, sa, sb, [32, 32], torch.bfloat16
        )
        == "common fp8"
    )
    assert calls[0] == (a, b, sa, sb, [32, 32], torch.bfloat16)
    assert (
        alias.w8a8_block_fp8_matmul_triton(
            a, b, sa, sb, [32, 32], torch.float16
        )[0]
        == "native"
    )
    handle.uninstall()
    assert all(getattr(alias, name) is fn for name, fn in originals.items())


def test_common_abi_mismatch_fails_before_rebinding(monkeypatch, tmp_path):
    p = fresh()
    alias, originals, _, _, _ = mock_reviewed_install(
        monkeypatch, tmp_path, p, abi=2
    )
    with pytest.raises(RuntimeError, match="FlagGems SM90 operator ABI"):
        p.install()
    assert all(getattr(alias, name) is fn for name, fn in originals.items())
    assert p._handle is None
