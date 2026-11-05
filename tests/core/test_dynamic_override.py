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

"""
Unit tests for dynamic operator override functionality.

Tests the DynamicOpOverride class from flaggems_sglang.dynamic_registry and
the CLI integration from flaggems_sglang.cli_override, verifying:

- Basic override and restore operations
- Loading implementations from files
- Batch override operations
- Context manager behavior
- CLI argument parsing and application
- pytest integration via --override and --override-config options

The real flaggems_sglang operators are Triton kernels that need a GPU, so
these tests override synthetic operator names (``_ovr_test_*``) with plain
CPU torch implementations. The override mechanism is pure ``setattr`` on the
package module, so it exercises the same code path regardless of whether the
target name was a real kernel; overriding names that don't exist yet is a
supported case (restore() removes them again).
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.dynamic_registry import DynamicOpOverride

# Needed for TestPytestIntegration, which uses the `pytester` fixture to run
# pytest itself as a subprocess and verify the --override/--override-config
# CLI options work end-to-end through the real pytest hooks.
pytest_plugins = ["pytester"]

# Synthetic op names used throughout; none collide with real operators, so
# the "override a name that does not exist yet" path (restore removes it) is
# what gets exercised.
OP_ABS = "_ovr_test_abs"
OP_NEG = "_ovr_test_neg"
OP_ADD = "_ovr_test_add"


# Test fixtures for custom implementations
def custom_abs_impl(input):
    """Custom implementation that adds a marker."""
    result = torch.abs(input)
    result._custom_marker = "custom_abs_called"
    return result


def custom_neg_impl(input):
    """Custom implementation of neg."""
    result = torch.neg(input)
    result._custom_marker = "custom_neg_called"
    return result


def custom_add_impl(input, other, *, alpha=1):
    """Custom implementation of add."""
    result = torch.add(input, other, alpha=alpha)
    result._custom_marker = "custom_add_called"
    return result


class TestDynamicOpOverride:
    """Test suite for DynamicOpOverride class."""

    def test_basic_override(self):
        """Test basic operator override functionality."""
        registry = DynamicOpOverride()

        success = registry.override(OP_ABS, custom_abs_impl)
        assert success, "Override should succeed"

        x = torch.tensor([-1.0, -2.0, 3.0])
        result = getattr(flaggems_sglang, OP_ABS)(x)

        assert hasattr(
            result, "_custom_marker"
        ), "Custom implementation should be called"
        assert result._custom_marker == "custom_abs_called"

        expected = torch.abs(x)
        torch.testing.assert_close(result, expected)

        registry.restore(OP_ABS)

    def test_get_override_tracks_replacement_and_restore(self):
        registry = DynamicOpOverride()
        assert registry.get_override(OP_ABS) is None
        assert registry.override(OP_ABS, custom_abs_impl)
        assert registry.get_override(OP_ABS) is custom_abs_impl
        assert registry.override(OP_ABS, custom_neg_impl)
        assert registry.get_override(OP_ABS) is custom_neg_impl
        registry.restore_all()
        assert registry.get_override(OP_ABS) is None

    def test_get_override_preserves_file_call_tracking(self, tmp_path):
        implementation = tmp_path / "candidate.py"
        implementation.write_text("def run(value): return value + 1\n")
        registry = DynamicOpOverride()
        assert registry.override_from_file(OP_ABS, str(implementation), "run")
        candidate = registry.get_override(OP_ABS)
        assert candidate is getattr(flaggems_sglang, OP_ABS)
        assert candidate(3) == 4
        # Calling through the public resolver must satisfy the same tracking
        # check as invoking the package attribute directly.
        registry.restore_all()
        assert registry.get_override(OP_ABS) is None

    def test_multiple_overrides(self):
        """Test overriding multiple operators."""
        registry = DynamicOpOverride()

        registry.override(OP_ABS, custom_abs_impl)
        registry.override(OP_NEG, custom_neg_impl)

        x = torch.tensor([1.0, -2.0, 3.0])

        abs_result = getattr(flaggems_sglang, OP_ABS)(x)
        assert abs_result._custom_marker == "custom_abs_called"

        neg_result = getattr(flaggems_sglang, OP_NEG)(x)
        assert neg_result._custom_marker == "custom_neg_called"

        registry.restore_all()

    def test_restore_single_op(self):
        """Test restoring a single overridden operator."""
        registry = DynamicOpOverride()

        # OP_ABS does not exist before the override; restore must remove it.
        assert not hasattr(flaggems_sglang, OP_ABS)

        registry.override(OP_ABS, custom_abs_impl)
        assert getattr(flaggems_sglang, OP_ABS) is custom_abs_impl

        success = registry.restore(OP_ABS)
        assert success, "Restore should succeed"
        assert not hasattr(
            flaggems_sglang, OP_ABS
        ), "Absent attribute should be removed on restore"

    def test_restore_all(self):
        """Test restoring all overridden operators."""
        registry = DynamicOpOverride()

        registry.override(OP_ABS, custom_abs_impl)
        registry.override(OP_NEG, custom_neg_impl)

        registry.restore_all()

        assert not hasattr(flaggems_sglang, OP_ABS)
        assert not hasattr(flaggems_sglang, OP_NEG)

    def test_context_manager(self):
        """Test using DynamicOpOverride as context manager."""
        with DynamicOpOverride() as registry:
            registry.override(OP_ABS, custom_abs_impl)

            x = torch.tensor([-1.0])
            result = getattr(flaggems_sglang, OP_ABS)(x)
            assert hasattr(result, "_custom_marker")

        # Should be restored after exiting context
        assert not hasattr(flaggems_sglang, OP_ABS)

    def test_list_overrides(self):
        """Test listing active overrides."""
        registry = DynamicOpOverride()

        assert len(registry.list_overrides()) == 0

        registry.override(OP_ABS, custom_abs_impl)
        overrides = registry.list_overrides()
        assert len(overrides) == 1
        assert f"flaggems_sglang.{OP_ABS}" in overrides

        registry.override(OP_NEG, custom_neg_impl)
        assert len(registry.list_overrides()) == 2

        registry.restore_all()
        assert len(registry.list_overrides()) == 0

    def test_override_from_file(self):
        """Test loading implementation from a file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                "import torch\n\n"
                "def my_custom_abs(input):\n"
                "    result = torch.abs(input)\n"
                '    result._file_marker = "loaded_from_file"\n'
                "    return result\n"
            )
            temp_file = f.name

        try:
            registry = DynamicOpOverride()

            success = registry.override_from_file(
                OP_ABS, temp_file, "my_custom_abs"
            )
            assert success, "Should load from file successfully"

            x = torch.tensor([-1.0, -2.0])
            result = getattr(flaggems_sglang, OP_ABS)(x)
            assert result._file_marker == "loaded_from_file"

            registry.restore_all()
        finally:
            Path(temp_file).unlink()

    def test_override_batch_from_files(self):
        """Test batch override from multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            abs_file = tmpdir / "custom_abs.py"
            abs_file.write_text(
                "import torch\n"
                "def my_abs(input):\n"
                "    result = torch.abs(input)\n"
                '    result._marker = "abs_from_file"\n'
                "    return result\n"
            )

            neg_file = tmpdir / "custom_neg.py"
            neg_file.write_text(
                "import torch\n"
                "def my_neg(input):\n"
                "    result = torch.neg(input)\n"
                '    result._marker = "neg_from_file"\n'
                "    return result\n"
            )

            registry = DynamicOpOverride()

            results = registry.override_batch_from_files(
                {
                    OP_ABS: (str(abs_file), "my_abs"),
                    OP_NEG: (str(neg_file), "my_neg"),
                }
            )

            assert results[OP_ABS], "abs override should succeed"
            assert results[OP_NEG], "neg override should succeed"

            x = torch.tensor([1.0, -2.0])

            abs_result = getattr(flaggems_sglang, OP_ABS)(x)
            assert abs_result._marker == "abs_from_file"

            neg_result = getattr(flaggems_sglang, OP_NEG)(x)
            assert neg_result._marker == "neg_from_file"

            registry.restore_all()

    def test_override_nonexistent_file(self):
        """Test handling of nonexistent file."""
        registry = DynamicOpOverride()

        success = registry.override_from_file(
            OP_ABS, "/nonexistent/path/file.py", "some_func"
        )

        assert not success, "Should fail for nonexistent file"

    def test_override_missing_function(self):
        """Test handling of missing function in file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def some_other_function():\n    pass\n")
            temp_file = f.name

        try:
            registry = DynamicOpOverride()

            success = registry.override_from_file(
                OP_ABS, temp_file, "nonexistent_function"
            )

            assert not success, "Should fail for missing function"
        finally:
            Path(temp_file).unlink()

    def test_restore_without_override(self):
        """Test restoring an operator that wasn't overridden."""
        registry = DynamicOpOverride()

        success = registry.restore(OP_ABS)
        assert not success, "Should fail when operator wasn't overridden"

    def test_operator_with_kwargs(self):
        """Test overriding operators with keyword arguments."""
        registry = DynamicOpOverride()

        registry.override(OP_ADD, custom_add_impl)

        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])

        result = getattr(flaggems_sglang, OP_ADD)(x, y, alpha=2.0)
        assert result._custom_marker == "custom_add_called"

        expected = x + 2.0 * y
        torch.testing.assert_close(result, expected)

        registry.restore_all()

    def test_concurrent_registries(self):
        """Test multiple independent registries share the same module."""
        registry1 = DynamicOpOverride()

        registry1.override(OP_ABS, custom_abs_impl)
        assert getattr(flaggems_sglang, OP_ABS) is custom_abs_impl

        # A second observer sees the override from registry1 (same module).
        x = torch.tensor([-1.0])
        result = getattr(flaggems_sglang, OP_ABS)(x)
        assert hasattr(result, "_custom_marker")

        registry1.restore(OP_ABS)
        assert not hasattr(flaggems_sglang, OP_ABS)

    def test_override_unimplemented_operator(self):
        """Overriding a name absent from the module, then restoring it."""
        registry = DynamicOpOverride()

        name = "_ovr_test_not_implemented"
        assert not hasattr(flaggems_sglang, name)

        def candidate(x):
            return x + 1

        assert registry.override(name, candidate)
        assert getattr(flaggems_sglang, name)(torch.tensor([1.0])) == 2.0

        registry.restore_all()
        assert not hasattr(flaggems_sglang, name)


class TestCLIOverride:
    """Test suite for CLI override functionality."""

    def test_parse_override_spec_colon(self):
        """Test parsing override spec with colon separator."""
        from flaggems_sglang.cli_override import parse_override_spec

        op_name, filepath, func_name = parse_override_spec(
            "silu_and_mul:./custom.py"
        )
        assert op_name == "silu_and_mul"
        assert filepath == "./custom.py"
        assert func_name == "silu_and_mul"

        op_name, filepath, func_name = parse_override_spec(
            "silu_and_mul:./custom.py:my_silu"
        )
        assert op_name == "silu_and_mul"
        assert filepath == "./custom.py"
        assert func_name == "my_silu"

    def test_parse_override_spec_equals(self):
        """Test parsing override spec with equals separator."""
        from flaggems_sglang.cli_override import parse_override_spec

        op_name, filepath, func_name = parse_override_spec(
            "silu_and_mul=./custom.py"
        )
        assert op_name == "silu_and_mul"
        assert filepath == "./custom.py"
        assert func_name == "silu_and_mul"

        op_name, filepath, func_name = parse_override_spec(
            "silu_and_mul=./custom.py:my_silu"
        )
        assert op_name == "silu_and_mul"
        assert filepath == "./custom.py"
        assert func_name == "my_silu"

    def test_parse_override_spec_invalid(self):
        """Test parsing invalid override spec."""
        from flaggems_sglang.cli_override import parse_override_spec

        with pytest.raises(ValueError):
            parse_override_spec("invalid_spec_without_separator")

    def test_load_override_config_yaml(self):
        """Test loading override config from YAML file."""
        from flaggems_sglang.cli_override import load_override_config

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(
                "overrides:\n"
                "  silu_and_mul:\n"
                "    file: ./custom_silu.py\n"
                "    function: my_silu\n"
                "  l2norm:\n"
                "    file: ./custom_l2norm.py\n"
                "  gelu_and_mul: ./custom_gelu.py\n"
                "  merge_state: ./custom_merge.py:my_merge\n"
            )
            temp_file = f.name

        try:
            overrides = load_override_config(temp_file)

            assert overrides["silu_and_mul"] == (
                "./custom_silu.py",
                "my_silu",
            )
            assert overrides["l2norm"] == ("./custom_l2norm.py", "l2norm")
            assert overrides["gelu_and_mul"] == (
                "./custom_gelu.py",
                "gelu_and_mul",
            )
            assert overrides["merge_state"] == (
                "./custom_merge.py",
                "my_merge",
            )
        finally:
            Path(temp_file).unlink()

    def test_load_override_config_json(self):
        """Test loading override config from JSON file."""
        from flaggems_sglang.cli_override import load_override_config

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(
                "{\n"
                '  "overrides": {\n'
                '    "silu_and_mul": {\n'
                '      "file": "./custom_silu.py",\n'
                '      "function": "my_silu"\n'
                "    },\n"
                '    "l2norm": "./custom_l2norm.py"\n'
                "  }\n"
                "}\n"
            )
            temp_file = f.name

        try:
            overrides = load_override_config(temp_file)

            assert overrides["silu_and_mul"] == (
                "./custom_silu.py",
                "my_silu",
            )
            assert overrides["l2norm"] == ("./custom_l2norm.py", "l2norm")
        finally:
            Path(temp_file).unlink()

    def test_load_override_config_missing_file(self):
        """Test loading config from nonexistent file."""
        from flaggems_sglang.cli_override import load_override_config

        with pytest.raises(FileNotFoundError):
            load_override_config("/nonexistent/config.yaml")

    def test_shared_configuration_is_applied_and_restored_once(self, tmp_path):
        from flaggems_sglang.cli_override import (
            cleanup_overrides,
            configure_overrides,
        )

        implementation = tmp_path / "candidate.py"
        implementation.write_text("def run(value): return value + 1\n")
        cleanups = []
        config = SimpleNamespace(
            option=SimpleNamespace(
                override=[f"{OP_ABS}:{implementation}:run"],
                override_config=None,
            ),
            add_cleanup=cleanups.append,
        )
        first = configure_overrides(config)
        candidate = first.get_override(OP_ABS)
        assert configure_overrides(config) is first
        assert first.get_override(OP_ABS) is candidate
        assert len(cleanups) == 1
        assert candidate(3) == 4
        cleanup_overrides(config)
        cleanup_overrides(config)
        cleanups[0]()
        assert first.get_override(OP_ABS) is None
        assert not hasattr(flaggems_sglang, OP_ABS)

    def test_shared_cleanup_rejects_unused_candidate_once(self, tmp_path):
        from flaggems_sglang.cli_override import (
            cleanup_overrides,
            configure_overrides,
        )

        implementation = tmp_path / "candidate.py"
        implementation.write_text("def run(value): return value + 1\n")
        cleanups = []
        config = SimpleNamespace(
            option=SimpleNamespace(
                override=[f"{OP_ABS}:{implementation}:run"],
                override_config=None,
            ),
            add_cleanup=cleanups.append,
        )
        configure_overrides(config)
        with pytest.raises(AssertionError, match="never invoked"):
            cleanup_overrides(config)
        # One conftest reports the failure; the second conftest and pytest's
        # fallback cleanup must not report it again or leave the override.
        cleanup_overrides(config)
        cleanups[0]()
        assert not hasattr(flaggems_sglang, OP_ABS)

    def test_failed_configuration_restores_partial_overrides(self, tmp_path):
        from flaggems_sglang.cli_override import configure_overrides

        implementation = tmp_path / "candidate.py"
        implementation.write_text("def run(value): return value + 1\n")
        cleanups = []
        config = SimpleNamespace(
            option=SimpleNamespace(
                override=[
                    f"{OP_ABS}:{implementation}:run",
                    f"{OP_NEG}:{tmp_path / 'missing.py'}:run",
                ],
                override_config=None,
            ),
            add_cleanup=cleanups.append,
        )
        with pytest.warns(UserWarning, match="Failed to load"):
            with pytest.raises(SystemExit) as error:
                configure_overrides(config)
        assert error.value.code == 1
        assert not hasattr(config, "_override_registry")
        assert not hasattr(flaggems_sglang, OP_ABS)
        assert not hasattr(flaggems_sglang, OP_NEG)
        cleanups[0]()


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_ab_testing_scenario(self):
        """Test A/B testing two different implementations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            impl_a = tmpdir / "impl_a.py"
            impl_a.write_text(
                "import torch\n"
                "def op_impl(input):\n"
                "    result = torch.abs(input)\n"
                '    result._variant = "A"\n'
                "    return result\n"
            )

            impl_b = tmpdir / "impl_b.py"
            impl_b.write_text(
                "import torch\n"
                "def op_impl(input):\n"
                "    result = torch.abs(input)\n"
                '    result._variant = "B"\n'
                "    return result\n"
            )

            x = torch.randn(10, 10)

            with DynamicOpOverride() as reg_a:
                reg_a.override_from_file(OP_ABS, str(impl_a), "op_impl")
                result_a = getattr(flaggems_sglang, OP_ABS)(x)
                assert result_a._variant == "A"

            with DynamicOpOverride() as reg_b:
                reg_b.override_from_file(OP_ABS, str(impl_b), "op_impl")
                result_b = getattr(flaggems_sglang, OP_ABS)(x)
                assert result_b._variant == "B"

            torch.testing.assert_close(result_a, result_b)

    def test_progressive_override(self):
        """Test progressively adding overrides during testing."""
        registry = DynamicOpOverride()

        x = torch.tensor([1.0, -2.0])

        registry.override(OP_ABS, custom_abs_impl)
        result = getattr(flaggems_sglang, OP_ABS)(x)
        assert hasattr(result, "_custom_marker")

        registry.override(OP_NEG, custom_neg_impl)
        result = getattr(flaggems_sglang, OP_NEG)(x)
        assert hasattr(result, "_custom_marker")

        registry.restore(OP_ABS)
        assert not hasattr(flaggems_sglang, OP_ABS)

        # OP_NEG is still overridden.
        result = getattr(flaggems_sglang, OP_NEG)(x)
        assert hasattr(result, "_custom_marker")

        registry.restore_all()


class TestPytestIntegration:
    """
    End-to-end tests that invoke pytest as a subprocess (via the
    ``pytester`` fixture) to verify that ``--override``/``--override-config``
    work through the real ``pytest_addoption``/``pytest_configure``/
    ``pytest_unconfigure`` hooks.

    These tests exist to catch integration bugs (e.g. option-parsing API
    mismatches between ``argparse`` and pytest's own ``Parser``) that unit
    tests calling ``DynamicOpOverride`` directly cannot see, since those
    never go through ``pytest_addoption``.
    """

    @staticmethod
    def _repo_src_path() -> str:
        return str(Path(__file__).resolve().parents[2] / "src")

    _CONFTEST = """
        from flaggems_sglang.cli_override import (
            add_override_arguments,
            cleanup_overrides,
            configure_overrides,
        )

        def pytest_addoption(parser):
            add_override_arguments(parser)
            add_override_arguments(parser)

        def pytest_configure(config):
            configure_overrides(config)
            configure_overrides(config)

        def pytest_unconfigure(config):
            cleanup_overrides(config)
            cleanup_overrides(config)
        """

    def test_override_option_registered(self, pytester):
        """--override/--override-config/--list-overrides show up in --help."""
        pytester.makeini(f"[pytest]\npythonpath = {self._repo_src_path()}\n")
        pytester.makeconftest(self._CONFTEST)

        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*--override=SPEC*"])
        result.stdout.fnmatch_lines(["*--override-config=PATH*"])
        result.stdout.fnmatch_lines(["*--list-overrides*"])

    def test_override_applied_through_cli(self, pytester):
        """A test calling flaggems_sglang.<op>() sees the overridden impl."""
        pytester.makeini(f"[pytest]\npythonpath = {self._repo_src_path()}\n")
        pytester.makeconftest(self._CONFTEST)

        custom_impl = pytester.makepyfile(
            custom_abs=(
                "import torch\n\n"
                "def my_abs(input):\n"
                "    result = torch.abs(input)\n"
                '    result._custom_marker = "from_cli_override"\n'
                "    return result\n"
            )
        )

        pytester.makepyfile(
            test_uses_override=(
                "import torch\n"
                "import flaggems_sglang\n\n"
                "def test_abs_is_overridden():\n"
                "    x = torch.tensor([-1.0, -2.0])\n"
                f"    result = getattr(flaggems_sglang, '{OP_ABS}')(x)\n"
                "    assert getattr(result, '_custom_marker', None)"
                ' == "from_cli_override"\n'
            )
        )

        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--override={OP_ABS}:{custom_impl}:my_abs",
            "test_uses_override.py",
        )
        result.assert_outcomes(passed=1)

    def test_override_config_file_applied_through_cli(self, pytester):
        """--override-config loads a YAML file and applies the overrides."""
        pytester.makeini(f"[pytest]\npythonpath = {self._repo_src_path()}\n")
        pytester.makeconftest(self._CONFTEST)

        custom_impl = pytester.makepyfile(
            custom_neg=(
                "import torch\n\n"
                "def my_neg(input):\n"
                "    result = torch.neg(input)\n"
                '    result._custom_marker = "from_config_file"\n'
                "    return result\n"
            )
        )

        config_file = pytester.makefile(
            ".yaml",
            overrides=(
                "overrides:\n"
                f"  {OP_NEG}:\n"
                f"    file: {custom_impl}\n"
                "    function: my_neg\n"
            ),
        )

        pytester.makepyfile(
            test_uses_config_override=(
                "import torch\n"
                "import flaggems_sglang\n\n"
                "def test_neg_is_overridden():\n"
                "    x = torch.tensor([1.0, -2.0])\n"
                f"    result = getattr(flaggems_sglang, '{OP_NEG}')(x)\n"
                "    assert getattr(result, '_custom_marker', None)"
                ' == "from_config_file"\n'
            )
        )

        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--override-config={config_file}",
            "test_uses_config_override.py",
        )
        result.assert_outcomes(passed=1)

    def test_no_override_leaves_default(self, pytester):
        """Without --override, the synthetic op stays absent."""
        pytester.makeini(f"[pytest]\npythonpath = {self._repo_src_path()}\n")
        pytester.makeconftest(self._CONFTEST)

        pytester.makepyfile(
            test_no_override=(
                "import flaggems_sglang\n\n"
                "def test_op_not_overridden():\n"
                f"    assert not hasattr(flaggems_sglang, '{OP_ABS}')\n"
            )
        )

        result = pytester.runpytest(
            "-p", "no:cacheprovider", "test_no_override.py"
        )
        result.assert_outcomes(passed=1)
