# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""W4A8 selection follows runner metadata and preserves measured schedules."""

import copy
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class HistoricalAdapterTests(unittest.TestCase):
    def setUp(self):
        name = "isolated_historical_w4a8"
        package = types.ModuleType(name)
        package.__path__ = [
            str(Path(__file__).parents[1] / "src/flaggems_sglang_plugin")
        ]
        self.patch = patch.dict(sys.modules, {name: package})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.adapter = importlib.import_module(name + ".w4a8_moe")

    def table(self, experts=48, stage="w13"):
        module = self.adapter
        points = module._points(experts)
        return [
            (
                0 if index == 0 else points[index - 1],
                point,
                copy.deepcopy(
                    module.BASELINE_TACTICS[f"{experts}/{stage}/{point}"]
                ),
            )
            for index, point in enumerate(points)
        ]

    def test_current_runner_without_historical_interface_is_not_rewritten(
        self,
    ):
        spec = types.SimpleNamespace(
            origin="/nonexistent/moe_runner/humming.py"
        )
        with patch.object(
            self.adapter.importlib.util, "find_spec", return_value=spec
        ):
            before = set(sys.modules)
            self.assertIsNone(self.adapter.install_if_supported())
            self.assertEqual(set(sys.modules), before)

    def test_historical_runner_is_installed_without_environment_flags(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "humming.py").write_text("qualified runner")
            spec = types.SimpleNamespace(origin=str(root / "humming.py"))
            with patch.object(
                self.adapter.importlib.util, "find_spec", return_value=spec
            ), patch.object(
                self.adapter,
                "_sha",
                return_value=self.adapter.SOURCE_PINS["humming.py"][0],
            ), patch.object(
                self.adapter, "install", return_value="installed"
            ) as install:
                self.assertEqual(
                    self.adapter.install_if_supported(), "installed"
                )
                install.assert_called_once_with()

    def test_w4a8_metadata_admission_preserves_a16_and_other_layouts(self):

        experts = 16
        metas = {}
        for stage, n, k in (("w13", 4608, 5120), ("w2", 5120, 2304)):
            metas[stage] = types.SimpleNamespace(
                num_experts=experts,
                shape_n=n,
                shape_k=k,
                pad_shape_n=0,
                pad_shape_k=0,
                input_scale_group_size=0,
                weight_scale_group_size=32,
                weight_scale_group_size_n=0,
                has_zero_point=False,
                has_bias=False,
                use_int_weight_scale=False,
                use_fused_e8m0_scale=True,
                use_packed_k_layout=False,
                a_dtype="float8e4m3",
                b_dtype="float4e2m1",
                c_dtype="bfloat16",
                as_dtype="float32",
                bs_dtype="float8e8m0",
                mma_type="wgmma",
                weight_scale_type="group",
                weight_scale_2_type="tensor",
            )
        layer = types.SimpleNamespace(
            _dsv4_mxfp4_backend="humming",
            _humming_uses_deepep_fp8_dispatch=False,
            humming_metas=metas,
        )
        runner = types.SimpleNamespace(
            num_experts=experts,
            global_num_experts=128,
            layer=layer,
            activation="silu",
            swiglu_limit=10.0,
            estimate_local_valid_shape_m=lambda ids, m: 32,
        )
        hidden = types.SimpleNamespace(
            shape=(16, 1024, 5120),
            dtype=torch.bfloat16,
            is_cuda=True,
            device="cuda:0",
            is_contiguous=lambda: True,
        )
        ids = types.SimpleNamespace(ndim=2, shape=(8, 3))
        counts = types.SimpleNamespace(
            shape=(16,),
            dtype=torch.int32,
            is_cuda=True,
            device="cuda:0",
            is_contiguous=lambda: True,
        )
        self.assertEqual(
            self.adapter._eligible(runner, hidden, ids, counts, 2, None),
            (runner, experts, 32),
        )
        del layer._humming_uses_deepep_fp8_dispatch
        self.assertIsNotNone(
            self.adapter._eligible(runner, hidden, ids, counts, 2, None)
        )
        layer._humming_uses_deepep_fp8_dispatch = True
        self.assertIsNone(
            self.adapter._eligible(runner, hidden, ids, counts, 2, None)
        )
        layer._humming_uses_deepep_fp8_dispatch = False
        metas["w13"].a_dtype = "bfloat16"
        self.assertIsNone(
            self.adapter._eligible(runner, hidden, ids, counts, 2, None)
        )
        metas["w13"].a_dtype = "float8e4m3"
        hidden.shape = (16, 512, 5120)
        self.assertIsNone(
            self.adapter._eligible(runner, hidden, ids, counts, 2, None)
        )

    def test_cached_configs_cover_every_measured_hint_with_fp32_accumulators(
        self,
    ):
        for experts in (16, 48):
            runner = types.SimpleNamespace()
            result = self.adapter._configs(runner, experts)
            self.assertIs(self.adapter._configs(runner, experts), result)
            self.assertEqual(
                result["compute_config"],
                {"gemm_type": "grouped_masked", "use_f16_accum": False},
            )
            for stage in ("w13", "w2"):
                table = result[stage + "_tuning_config"]
                for hint in self.adapter._points(experts):
                    (tactic,) = [
                        value for lo, hi, value in table if lo < hint <= hi
                    ]
                    expected = copy.deepcopy(
                        self.adapter.BASELINE_TACTICS[
                            f"{experts}/{stage}/{hint}"
                        ]
                    )
                    expected["use_tma_c"] = False
                    self.assertEqual(tactic, expected)

    def test_supported_w4a8_executes_both_gemms_without_native_experiment_branches(
        self,
    ):

        for hint in (16, 32):
            events = []
            experts, capacity, width, intermediate = 16, 4, 8, 3
            hidden = torch.arange(
                experts * capacity * width, dtype=torch.float32
            ).reshape(experts, capacity, width)
            counts = torch.arange(experts).clamp_max(capacity).int()
            rows = experts * capacity
            buffers = {
                "quanted_gate_up_input": torch.empty(rows, width),
                "gate_up_output": torch.empty(rows, 2 * intermediate),
                "activation_output": torch.empty(rows, intermediate),
                "quanted_down_input": torch.empty(rows, intermediate),
                "down_output": torch.empty(experts, capacity, width),
            }
            runner = types.SimpleNamespace(layer=object())

            def prepare(**kwargs):
                events.append("buffers")
                self.assertEqual(kwargs["hidden_states"].shape, (rows, width))
                return buffers

            runner.prepare_buffers = prepare

            def quant(inputs, output, scales, mask, **kwargs):
                self.assertIs(mask, counts)
                events.append("native-quant" if kwargs else "common-quant")
                self.assertEqual(scales.shape, (rows, 1))
                output.copy_(inputs.reshape_as(output))
                scales.fill_(1)
                return output, scales

            def gemm(**kwargs):
                events.append(kwargs["sublayer_name"])
                self.assertEqual(kwargs["valid_shape_m"], hint)
                self.assertIs(kwargs["expert_layout"], counts)
                kwargs["outputs"].fill_(2)

            def activate(inputs, output, mask, **kwargs):
                events.append("activation")
                self.assertIs(mask, counts)
                output.fill_(3)

            out = self.adapter._run_w4a8(
                runner,
                hidden,
                torch.zeros(1, 3),
                counts,
                (runner, experts, hint),
                torch=torch,
                method=types.SimpleNamespace(forward_layer=gemm),
                gemm_type="masked",
                compact_quant=lambda inputs, output, scales, mask: quant(
                    inputs, output, scales, mask, config="compact16w8"
                ),
                activate=activate,
                quant=lambda inputs, output, scales, mask: quant(
                    inputs, output, scales, mask
                ),
            )
            self.assertIs(out, buffers["down_output"])
            quant_event = "native-quant" if hint == 16 else "common-quant"
            self.assertEqual(
                events,
                [
                    "buffers",
                    quant_event,
                    "w13",
                    "activation",
                    quant_event,
                    "w2",
                ],
            )
            self.assertTrue((out == 2).all())

    def test_table_points_keep_interval_geometry_and_input_immutable(self):
        for experts in (16, 48):
            for stage in ("w13", "w2"):
                table = self.table(experts, stage)
                original = copy.deepcopy(table)
                result = self.adapter.tune_table(table, experts, stage)
                self.assertEqual(table, original)
                for point in self.adapter._points(experts):
                    selected = [v for lo, hi, v in result if lo < point <= hi]
                    self.assertEqual(len(selected), 1)
                    expected = copy.deepcopy(
                        self.adapter.BASELINE_TACTICS[
                            f"{experts}/{stage}/{point}"
                        ]
                    )
                    expected["use_tma_c"] = False
                    self.assertEqual(selected[0], expected)

    def test_unreviewed_native_tactic_rejected(self):
        table = self.table()
        table[0][2]["num_stages"] += 1
        with self.assertRaisesRegex(ValueError, "Unreviewed native"):
            self.adapter.tune_table(table, 48, "w13")

    def test_uninstall_keeps_later_binding_owner(self):
        old, owned, later = object(), object(), object()
        module = types.SimpleNamespace(first=owned, second=later)
        handle = self.adapter.Installation(
            [(module, "first", old, owned), (module, "second", old, owned)]
        )
        handle.uninstall()
        self.assertIs(module.first, old)
        self.assertIs(module.second, later)


if __name__ == "__main__":
    unittest.main()
