<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Contributing a New Operator

Thanks for your interest in `flaggems_sglang`. This guide walks through the
full lifecycle of shipping a new operator: deciding where it lives, wiring it
into the multi-level dispatcher, adding tests and benchmarks, meeting the code
style requirements, and getting it through CI.

If you're integrating a whole new hardware vendor rather than a single
operator, start with
[`src/flaggems_sglang/runtime/backend/README.md`](../src/flaggems_sglang/runtime/backend/README.md)
first — it covers vendor bring-up. Come back here once you have a vendor
folder in place.

---

## 1. Pick the right layer

Operators are resolved through a three-level registrar
(`runtime/op_registrar.py`). Later levels override earlier ones on name
collision:

| Priority | Location                                            | Purpose                       |
|----------|-----------------------------------------------------|-------------------------------|
| 0        | `src/flaggems_sglang/ops/`                          | Generic Triton fallback       |
| 1        | `src/flaggems_sglang/runtime/backend/_<vendor>/ops/`| Per-vendor specialization     |
| 2        | `src/flaggems_sglang/runtime/backend/_<vendor>/<arch>/ops/` | Per-arch specialization |

Use this decision tree:

- **New op that any Triton-capable device can run** → put it under
  `src/flaggems_sglang/ops/`. This is the default.
- **Vendor-specific rewrite of an existing op** (e.g. leveraging a vendor
  intrinsic, or working around a compiler quirk) → put it under
  `runtime/backend/_<vendor>/ops/`.
- **Arch-specific rewrite** (e.g. Hopper WGMMA path vs. Ampere) → put it
  under `runtime/backend/_<vendor>/<arch>/ops/`.

Routing is by **function name**, not import path. A vendor override for
an op simply defines `def my_op(...)` in
`_<vendor>/ops/my_op.py` and re-exports it via that folder's
`ops/__init__.py`. Missing operators fall back to the generic implementation
automatically, so vendors only ship what they specialize.

---

## 2. Add the operator

The examples below assume a new op called `my_op`.

### 2.1 Generic implementation

Create `src/flaggems_sglang/ops/my_op.py`:

```python
# Copyright 2026 FlagOS Contributors
# ... (Apache 2.0 header — see any existing file)

import torch
import triton
import triton.language as tl


@triton.jit
def _my_op_kernel(...):
    ...


def my_op(x: torch.Tensor, ...) -> torch.Tensor:
    """One-line summary. Extend if the shape/dtype contract is non-obvious."""
    ...
    return out


__all__ = ["my_op"]
```

Rules:

- **Every op file must define `__all__`.** The registrar walks `__all__` to
  discover public entry points; anything not listed is invisible.
- **File name should match the primary op name** (`my_op.py` exports `my_op`).
  It is what CI's `tools/select_tests.py` uses to auto-select the matching
  test and benchmark files.
- **Include the Apache 2.0 copyright header** at the top of every source
  file. Copy it verbatim from an existing file.
- Prefer `torch.Tensor` type hints on the public signature; keep the
  Triton kernel as a private `_` -prefixed function.
- Do not import from `runtime.backend.*` inside a generic op — the whole
  point of the generic tier is that it's device-agnostic.

### 2.2 Vendor specialization

Create `src/flaggems_sglang/runtime/backend/_<vendor>/ops/my_op.py` with the
same function name and the same `__all__ = ["my_op"]` contract as a generic
op. The registrar walks every `*.py` in the vendor `ops/` package the same
way it walks the generic tier — you do **not** need to re-export from
`_<vendor>/ops/__init__.py`.

### 2.3 Arch specialization

Same idea, one level deeper:
`_<vendor>/<arch>/ops/my_op.py`. `<arch>` is resolved via the vendor's
`ARCH_MAP` (see `_nvidia/__init__.py`). Again: the op file's `__all__` is
what gets discovered; the `ops/__init__.py` at this tier does not need any
re-export.

### 2.4 Verify the registration

After the op is in place, from a Python shell:

```python
import flaggems_sglang
assert "my_op" in flaggems_sglang.all_registered_ops()
print(flaggems_sglang.my_op.__module__)  # confirms which tier resolved
```

---

## 3. Add tests

Every new op needs a correctness test at `tests/test_my_op.py`. The filename
must match the source stem so CI picks it up automatically.

Skeleton — mirror an existing test like `tests/test_silu_and_mul.py`:

```python
# Copyright 2026 FlagOS Contributors
# ... (Apache 2.0 header)

import pytest
import torch

import flaggems_sglang
from . import conftest as cfg

FLOAT_DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 512), (4, 1024), (32, 2048)]


def _ref_my_op(x, ...):
    """Reference implementation — prefer sgl_kernel or a torch equivalent."""
    ...


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.my_op
def test_my_op(shape, dtype):
    device = cfg.device
    x = torch.randn(*shape, dtype=dtype, device=device)

    ref = _ref_my_op(x)
    res = flaggems_sglang.my_op(x)

    atol = 1e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(res, ref, atol=atol, rtol=1e-2)
```

Guidelines:

- **Reference correctness against an established implementation** — prefer
  `sgl_kernel` when available, otherwise
  a torch reference. Do not compare against your own kernel.
- **Tag each test with `@pytest.mark.<op_name>`.** Markers are used to
  select subsets in CI and locally.
- **Cover the dtypes and shape regimes the op actually targets.** For hidden
  dims, at minimum cover a small, a medium, and a large shape.
- **Tolerances should match dtype.** `1e-2` for fp16, `5e-3` for bf16 is a
  reasonable default for norm-shaped ops; adjust per numerics.
- **Do not gate on GPU-specific features** in the generic test — the same
  test runs across every vendor's runner.

Run it locally:

```shell
pytest -q tests/test_my_op.py --quick
```

---

## 4. Add a benchmark

Every new op also gets a perf benchmark at `benchmark/test_my_op.py`. Mirror
`benchmark/test_silu_and_mul.py`:

```python
# Copyright 2026 FlagOS Contributors
# ... (Apache 2.0 header)

import pytest
import torch

import flaggems_sglang
from .attri_util import FLOAT_DTYPES, MY_OP_BENCH_SHAPES


@pytest.mark.parametrize("shape", MY_OP_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.my_op
def test_my_op(shape, dtype, benchmark):
    device = flaggems_sglang.device
    x = torch.randn(*shape, dtype=dtype, device=device)
    benchmark(flaggems_sglang.my_op, x)
```

Add your shape list (`MY_OP_BENCH_SHAPES`) to `benchmark/attri_util.py`, next
to the existing per-op shape constants. Pick shapes representative of the
serving scenarios the op is designed for (decode vs. prefill, typical hidden
dims for target models).

Run locally:

```shell
pytest -q benchmark/test_my_op.py --level core --iter 1 --warmup 1
```

---

## 5. Code style

The full style contract lives in `.pre-commit-config.yaml` and `.flake8`.
Summary:

- **Formatter: `black`, line length 79.**
- **Import sort: `isort` with `--profile black`, line length 80.**
- **Linter: `flake8`, max line 79, extend-ignore `E203`** (pre-commit also
  ignores `F405,E731,W503,E203,E704` at 120 cols — but keep new code within
  the 79-col black limit).
- **Trailing whitespace and end-of-file newlines** are enforced by
  pre-commit hooks.
- **Apache 2.0 copyright header** is required on every new `.py`, `.yaml`,
  and `.md` file. Copy it from an existing file — the CI style job will
  fail otherwise.

Install and run pre-commit once per clone:

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Additional conventions from the existing codebase:

- Public function signatures use `torch.Tensor` type hints where meaningful.
- Prefix Triton `@triton.jit` kernels with `_` — they are implementation
  detail, not part of the public API.
- Docstrings are short; add a paragraph only when the tuning story or an
  invariant would surprise a future reader.
- Don't add comments that describe *what* the code does — well-named
  identifiers already do that. Add a comment only when the *why* is
  non-obvious (a hidden constraint, a workaround, a chosen tradeoff).

---

## 6. CI

CI runs on every push and PR to `master` (`.github/workflows/basic-ci.yml`).
There are three stages:

1. **`code-style`** — runs `pre-commit run --all-files` on Ubuntu. Fails on
   any formatter, linter, or copyright-header issue. Passing this locally
   before pushing saves a round trip.
2. **`select-targets`** — runs `tools/select_tests.py` against the diff to
   pick the minimal set of tests and benchmarks impacted by your change.
   The selector matches source files to tests **by stem**
   (`src/.../ops/my_op.py` → `tests/test_my_op.py` and
   `benchmark/test_my_op.py`). Sticking to the naming convention above is
   what makes automatic selection work.
3. **`nvidia-tests`** — runs the selected tests (`pytest --quick`) and
   benchmarks (`pytest --level core`) on the NVIDIA runner. Benchmark
   output is posted back to the PR.

Additional runners (AMD, Ascend, Iluvatar, Kunlunxin, Metax, Mthreads,
Hygon, T-Head) may be attached to your PR depending on vendor coverage. If
your change is vendor-specific, note in the PR description which runners
should exercise it.

**How to keep CI happy:**

- Keep filenames aligned (`my_op.py` ↔ `test_my_op.py` ↔
  `benchmark/test_my_op.py`).
- Don't add heavy imports at module top-level that require a specific
  vendor SDK — CI style runs on plain Ubuntu with no GPU.
- New pytest markers (`@pytest.mark.my_op`) don't need registration;
  `pytest.ini` sets `filterwarnings = ignore::pytest.PytestUnknownMarkWarning`.
- If your op requires a new Python dependency, add it to `pyproject.toml`
  and call it out in the PR description — CI runners have preconfigured
  venvs that may need refreshing.

---

## 7. Opening the PR

Use the template at `.github/PULL_REQUEST_TEMPLATE.md`. Fill in:

- **Header** - Each file must contain a copyright notice comment. The last line "Generated by XXX" is optional.
```
# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Generated by XXX
```
- **PR Category** — pick from `Operator | OP Test | Model Test | Benchmark |
  CI/CD | User Experience | Other`. A new op typically touches
  `Operator + OP Test + Benchmark`.
- **Type of Change** — `New Feature`, `Performance Optimization`, etc.
- **Description** — what the op does, what tier it targets (generic / vendor
  / arch), and any calling convention notes.
- **Issue** — link the tracking issue if there is one.
- **Progress checklist** — tests, review, coverage.
- **Performance** — if it's a perf change, paste the before/after benchmark
  numbers for the shapes your op cares about.

Reviewer expectations:

- At least 1 reviewer; 2 recommended.
- Every change is covered by at least one unit test.
- For perf work, include benchmark deltas in the PR body.
- Push to a topic branch, not `master`.

---

## 8. Quick checklist

Before you hit "Create PR":

- [ ] Op file has Apache 2.0 header and defines `__all__`.
- [ ] File name matches op name (`my_op.py` exports `my_op`).
- [ ] Op appears in `flaggems_sglang.all_registered_ops()` locally.
- [ ] `tests/test_my_op.py` exists, references a trusted baseline, is
      tagged with `@pytest.mark.my_op`.
- [ ] `benchmark/test_my_op.py` exists with representative shapes added to
      `benchmark/attri_util.py`.
- [ ] `pre-commit run --all-files` passes.
- [ ] `pytest -q tests/test_my_op.py --quick` passes on your device.
- [ ] For vendor/arch specializations: op file defines
      `__all__ = ["<op_name>"]`.
- [ ] PR description follows the template.

---

## 9. FlagOS × SGLang competition submissions

Submissions for the **FlagOS × SGLang** operator competition follow a
lighter-weight process than a full contribution — competitors ship the
operator only, and the maintainers handle correctness and performance
validation against a held-out harness.

**What you submit:**

- The operator source file, in the correct tier of the dispatcher tree.
  Same rules as §1 and §2:
  - Generic Triton kernel → `src/flaggems_sglang/ops/<op_name>.py`.
  - Vendor-specialized → `src/flaggems_sglang/runtime/backend/_<vendor>/ops/<op_name>.py`.
  - Arch-specialized → `src/flaggems_sglang/runtime/backend/_<vendor>/<arch>/ops/<op_name>.py`.
- Every op file defines `__all__ = ["<op_name>"]`. The registrar picks up
  every tier by walking the corresponding `ops/` package and reading each
  module's `__all__` — no `ops/__init__.py` re-export needed at any tier.
- The Apache 2.0 copyright header on every new file.

**What you do *not* need to submit:**

- No `tests/test_<op_name>.py`.
- No `benchmark/test_<op_name>.py` or shape constants in
  `benchmark/attri_util.py`.
- No PR-description performance numbers — the competition harness collects
  them centrally.

**What still must pass:**

Even without tests and benchmarks, competition PRs go through the same
`basic-ci.yml` pipeline. That means:

- **Signing the CLA is mandatory in order to submit code to this repository.**
- **`code-style` must pass.** Run `pre-commit run --all-files` locally
  before pushing — the Apache header, `black` (line 79), `isort`
  (`--profile black`), and `flake8` checks are all mandatory.
- **The op must import cleanly.** After your change,
  `import flaggems_sglang` should succeed and
  `flaggems_sglang.all_registered_ops()` should list your op. If it
  doesn't, `select-targets` won't pick anything up and the maintainers
  can't score your submission.
- **`select-targets` and downstream jobs must not error out.** Since you
  aren't adding a `test_<op>.py`, the selector will report no impacted
  tests and the vendor test job will be skipped — that is expected and
  fine. What must *not* happen is a syntax error or an import failure at
  package load time, which would fail every job.

**Competition-flavored PR description:**

Use the standard template at `.github/PULL_REQUEST_TEMPLATE.md`, and in the
**Description** section clearly state:

- That this is a **FlagOS × SGLang competition** submission.
- The **operator name** and the **tier** it targets
  (generic / vendor `_<vendor>` / arch `_<vendor>/<arch>`).
- The **target hardware** (vendor + arch) it was tuned for.
- Any calling-convention or dtype constraints the scoring harness needs
  to know about.
- **Accuracy Tests** is optional
- **Speed Tests and Profiling** results are from the submission system. You just copy the results and paste them here.

**Competition checklist:**

- [ ] Op file lives in the correct dispatcher tier folder.
- [ ] Apache 2.0 header + `__all__` present.
- [ ] For vendor/arch tiers: op file defines
      `__all__ = ["<op_name>"]`.
- [ ] `import flaggems_sglang; flaggems_sglang.all_registered_ops()`
      shows your op locally.
- [ ] `pre-commit run --all-files` passes.
- [ ] PR description tags this as a competition submission and names the
      target hardware.

---

Questions or gaps in this guide? Open an issue or ping a maintainer — this
document should stay in sync with the code, so improvements are welcome.
