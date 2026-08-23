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

import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "softcap_out.py"
)
SPEC = importlib.util.spec_from_file_location(
    "softcap_out_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class SoftcapOutTest(unittest.TestCase):
    def test_contiguous_fp32_matches_reference(self):
        x = torch.tensor(
            [-1500.0, -30.0, -1e-4, 0.0, 1e-4, 30.0, 1500.0],
            device="cuda",
            dtype=torch.float32,
        )
        cap = 30.0

        actual = MODULE.softcap_out(x, cap)
        expected = torch.tanh(x.to(torch.float32) / cap) * cap

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_large_cap_preserves_small_values(self):
        x = torch.tensor([-1.0, -0.25, 0.25, 1.0], device="cuda")
        cap = 1_000_000.0

        actual = MODULE.softcap_out(x, cap)
        expected = torch.tanh(x.to(torch.float32) / cap) * cap

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_numerical_boundaries_match_reference(self):
        normalized = torch.tensor(
            [
                0.0,
                -(2.0**-24),
                2.0**-24,
                -(2.0**-16),
                2.0**-16,
                -(2.0**-10),
                2.0**-10,
                -0.249999,
                0.249999,
                -0.25,
                0.25,
                -1.0,
                1.0,
                -10.0,
                10.0,
                -100.0,
                100.0,
            ],
            device="cuda",
        )
        for cap in (-30.0, -1.0, 0.5, 1.0, 30.0, 100.0, 1_000_000.0):
            with self.subTest(cap=cap):
                x = normalized * cap
                actual = MODULE.softcap_out(x, cap)
                expected = torch.tanh(x.to(torch.float32) / cap) * cap

                torch.testing.assert_close(
                    actual, expected, atol=1e-4, rtol=1e-4
                )

        special_x = torch.tensor(
            [
                float("-inf"),
                -1.0,
                -0.0,
                0.0,
                1.0,
                float("inf"),
                float("nan"),
            ],
            device="cuda",
        )
        for cap in (
            30.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            with self.subTest(special_cap=cap):
                actual = MODULE.softcap_out(special_x, cap)
                expected = torch.tanh(special_x.to(torch.float32) / cap) * cap

                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-4,
                    rtol=1e-4,
                    equal_nan=True,
                )

    def test_seeded_random_inputs_match_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        inputs = (
            torch.randn(4099, generator=generator, device="cuda") * 40.0,
            torch.rand(4099, generator=generator, device="cuda") * 200.0
            - 100.0,
        )

        for x in inputs:
            actual = MODULE.softcap_out(x, 30.0)
            expected = torch.tanh(x.to(torch.float32) / 30.0) * 30.0

            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_dtypes_and_tail_lengths_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        for dtype, tolerance in tolerances.items():
            for length in (
                1,
                17,
                63,
                64,
                65,
                127,
                128,
                129,
                255,
                256,
                257,
                511,
                512,
                513,
                1023,
                1024,
                1025,
            ):
                with self.subTest(dtype=dtype, length=length):
                    x = torch.linspace(
                        -60.0,
                        60.0,
                        length,
                        device="cuda",
                        dtype=dtype,
                    )
                    original = x.clone()

                    actual = MODULE.softcap_out(x, 30.0)
                    expected = torch.tanh(x.to(torch.float32) / 30.0) * 30.0

                    self.assertEqual(
                        (actual.shape, actual.dtype),
                        (x.shape, torch.float32),
                    )
                    torch.testing.assert_close(x, original, atol=0.0, rtol=0.0)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=tolerance,
                        rtol=tolerance,
                    )

    def test_empty_input_returns_empty_fp32(self):
        x = torch.empty((0, 3), device="cuda", dtype=torch.float16)

        actual = MODULE.softcap_out(x, 30.0)

        self.assertEqual(
            (actual.shape, actual.dtype), (x.shape, torch.float32)
        )

    def test_noncontiguous_input_matches_reference(self):
        base = torch.arange(18, device="cuda", dtype=torch.float32).reshape(
            3, 6
        )
        x = base[:, ::2]

        actual = MODULE.softcap_out(x, 5.0)
        expected = torch.tanh(x.to(torch.float32) / 5.0) * 5.0

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_scalar_tensor_cap_matches_reference(self):
        x = torch.tensor([-10.0, 0.0, 10.0], device="cuda")
        cap = torch.tensor(5.0, device="cuda")

        actual = MODULE.softcap_out(x, cap)
        expected = torch.tanh(x.to(torch.float32) / cap) * cap

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_cap_edge_cases_match_reference(self):
        x = torch.tensor([-1.0, 0.0, 1.0], device="cuda")
        caps = (
            0.0,
            -30.0,
            1,
            float.fromhex("0x1p-149"),
            float.fromhex("0x1p-128"),
            float.fromhex("0x1.000008p-128"),
            float.fromhex("0x1.fffffep+127"),
            torch.tensor(5, dtype=torch.int32),
        )

        for cap in caps:
            with self.subTest(cap=cap):
                actual = MODULE.softcap_out(x, cap)
                expected = torch.tanh(x.to(torch.float32) / cap) * cap

                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-4,
                    rtol=1e-4,
                    equal_nan=True,
                )

    def test_multiple_value_cap_is_rejected(self):
        cap = torch.tensor([5.0, 30.0], device="cuda")

        with self.assertRaisesRegex(ValueError, "one value"):
            MODULE.softcap_out(torch.ones(1, device="cuda"), cap)


if __name__ == "__main__":
    unittest.main()
