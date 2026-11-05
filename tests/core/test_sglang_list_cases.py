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

"""Real pytest-hook coverage for listing, selection and invalid catalogs."""

import json
from pathlib import Path

import pytest
import torch

pytest_plugins = ["pytester"]


@pytest.fixture
def listing_project(pytester):
    root = Path(__file__).resolve().parents[2]
    pytester.makeini(f"[pytest]\npythonpath = . {root / 'src'}\n")
    directory = pytester.path / "benchmark"
    directory.mkdir()
    for path in (root / "benchmark").iterdir():
        if path.is_file() and (
            not path.name.startswith("test_")
            or path.name == "test_silu_and_mul.py"
        ):
            (directory / path.name).symlink_to(path)
    pytester.makeconftest(
        """
        import pytest
        import torch

        @pytest.fixture(autouse=True)
        def forbid_tensor_building_during_listing(request, monkeypatch):
            if request.config.getoption("list_cases", False):
                def forbidden(*args, **kwargs):
                    raise AssertionError("listing allocated a Tensor")
                for name in ("randn", "rand", "zeros", "ones", "empty", "tensor"):
                    monkeypatch.setattr(torch, name, forbidden)
    """
    )
    return directory


@pytest.mark.parametrize("level,count", [("core", 5), ("comprehensive", 15)])
def test_actual_silu_lists_all_cases_without_building(
    pytester, listing_project, level, count
):
    output = pytester.path / "cases.json"
    result = pytester.runpytest_subprocess(
        "benchmark/test_silu_and_mul.py",
        "--list-cases",
        f"--level={level}",
        f"--output={output}",
        "-q",
        timeout=90,
    )
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    catalog = json.loads(output.read_text())
    assert catalog["schema_version"] == "flaggems.benchmark-case-list/v2"
    cases = catalog["benchmarks"][0]["cases"]
    assert len(cases) == len({case["case_id"] for case in cases}) == count
    for case in cases:
        assert case["dtype"] == "torch.bfloat16"
        params = case["params"]
        assert case["shape"] == {
            "hidden_states": [params["bs"], 2 * params["d"]]
        }
        assert params["seed"] == 0


def _write_parametrized_provider(directory, *, duplicate=False):
    code = """
import json
from pathlib import Path
import pytest
import torch
from benchmark.cases import BenchmarkCasePlan
from benchmark.op_benchmark import OpBenchmark

def cases(shape, dtype, scale):
    yield BenchmarkCasePlan(shape={"x": shape}, params={"scale": scale},
                            builder_args=(shape, scale))

def build(plan, dtype, device):
    with Path("built.jsonl").open("a") as output:
        output.write(json.dumps(plan.params) + "\\n")
    shape, scale = plan.builder_args
    return torch.ones(shape, dtype=dtype), scale

@pytest.mark.listing_probe
@pytest.mark.parametrize("scale", [2, 3], ids=["first", "second"])
def test_outer(scale):
    def run():
        bench = OpBenchmark(op_name="listing_probe", torch_op=lambda x,s: x*s,
            shapes=[(4,)], dtypes=[torch.float32], shape_desc="n",
            case_fn=lambda shape,dtype: cases(shape,dtype,scale),
            build_inputs_fn=build)
        bench.set_gems(lambda x,s: x*s)
        bench.run()
    run()
    __DUPLICATE__
"""
    path = directory / "test_outer.py"
    path.write_text(
        code.replace("__DUPLICATE__", "run()" if duplicate else "")
    )
    return "benchmark/test_outer.py"


def test_outer_parameters_keep_separate_ids_and_exact_replay(
    pytester, listing_project
):
    path = _write_parametrized_provider(listing_project)
    output = pytester.path / "cases.json"
    result = pytester.runpytest_subprocess(
        path,
        "--list-cases",
        "--level=core",
        f"--output={output}",
        "-q",
        timeout=90,
    )
    result.assert_outcomes(passed=2)
    assert result.ret == 0
    cases = [
        c
        for b in json.loads(output.read_text())["benchmarks"]
        for c in b["cases"]
    ]
    assert [c["params"]["scale"] for c in cases] == [2, 3]
    assert len({c["case_id"] for c in cases}) == 2
    assert not (pytester.path / "built.jsonl").exists()

    replay = pytester.runpytest_subprocess(
        path,
        "--level=core",
        f"--case-id={cases[1]['case_id']}",
        "--metrics=latency",
        "--mode=operator",
        "--warmup=0",
        "--iter=1",
        "-q",
        timeout=90,
    )
    replay.assert_outcomes(passed=2)
    assert replay.ret == 0
    assert [
        json.loads(x)
        for x in (pytester.path / "built.jsonl").read_text().splitlines()
    ] == [{"scale": 3}]

    unknown = pytester.runpytest_subprocess(
        path,
        "--level=core",
        "--case-id=not-a-case",
        "-q",
        timeout=90,
    )
    assert unknown.ret != 0
    assert [
        json.loads(x)
        for x in (pytester.path / "built.jsonl").read_text().splitlines()
    ] == [{"scale": 3}]


def test_repeated_catalog_ids_fail_in_real_pytest(pytester, listing_project):
    path = _write_parametrized_provider(listing_project, duplicate=True)
    result = pytester.runpytest_subprocess(
        path,
        "--list-cases",
        "--level=core",
        "-q",
        timeout=90,
    )
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*duplicate case IDs*"])
    assert not (pytester.path / "built.jsonl").exists()


def test_silu_builder_preserves_original_seed_and_values():
    from benchmark.test_silu_and_mul import _build_inputs, _case_fn

    shape = (3, 7)
    dtype = torch.bfloat16
    (plan,) = _case_fn(shape, dtype)
    generator = torch.Generator(device="cpu").manual_seed(0)
    original = torch.randn(
        3, 14, dtype=torch.float32, device="cpu", generator=generator
    ).to(dtype)
    (actual,) = _build_inputs(plan, dtype, "cpu")
    assert actual.shape == original.shape
    assert actual.stride() == original.stride()
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
