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

# Multi-backend adaptation

This README is the entry point for **hardware vendors** integrating their
own backend into `flaggems_sglang`. If you're contributing a single
operator (generic, vendor, or arch tier) rather than bringing up a whole
new vendor, use [`docs/CONTRIBUTING.md`](../../../../docs/CONTRIBUTING.md)
instead — this file focuses on the vendor-bring-up flow.

## Introduction

`flaggems_sglang` supports multiple hardware backends through a
three-tier dispatcher rooted at
`src/flaggems_sglang/runtime/backend/`. To land your backend on the
official main branch, follow the steps below.

## Vendor bring-up

### Step 1 — create the vendor folder

Create a folder named after your vendor at
`src/flaggems_sglang/runtime/backend/`, following the pattern
`_<vendorname>`. Refer to
`src/flaggems_sglang/runtime/backend/_nvidia/` as the canonical example.

### Step 2 — populate the folder

Minimum layout:

```
_<vendor>/
├── __init__.py
└── ops/
    ├── __init__.py
    ├── add.py
    └── gelu.py
```

If your backend has multiple architectures (e.g. NVIDIA Hopper vs.
Ampere), add per-arch subfolders alongside `ops/`:

```
_<vendor>/
├── __init__.py
├── ops/                      # vendor-wide implementations
│   └── ...
├── hopper/                   # arch-specific specialization
│   ├── __init__.py
│   └── ops/
│       ├── __init__.py
│       └── my_op.py
└── ampere/
    ├── __init__.py
    └── ops/
        ├── __init__.py
        └── my_op.py
```

#### Step 2.1 — `__init__.py`

Copy `_nvidia/__init__.py` as a starting point. The **only change**
required is the `VendorDescriptor` construction:

```python
from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="xxx",
    device_name="xxx",
    device_query_cmd="xxx",
)
```

##### Required fields

- `vendor_name` — your vendor name (e.g. `nvidia`).
- `device_name` — your PyTorch device name (e.g. `cuda`).
- `device_query_cmd` — a shell command that only succeeds on your
  vendor's device (e.g. `nvidia-smi`). This is how the runtime
  auto-detects which backend to load.

##### Optional fields

- `dispatch_key` — `torch.library.Library` registration key (e.g.
  `PrivateUse1`).
- `triton_extra_name` — Triton extra module name (e.g. `hip`, `xpu`,
  `cann`).
- `fp64_enabled` / `bf16_enabled` / `int64_enabled` — dtype capability
  flags (default `True`).
- `tle_enabled` — whether the vendor exposes a TLE runtime hook
  (default `False`).

##### Arch mapping (optional)

If you ship arch-specific specializations, define `ARCH_MAP` at the
bottom of `_<vendor>/__init__.py`. Keys are the arch discriminator
strings returned by your device query (major compute capability, arch
family, etc.); values are subfolder names. Example from `_nvidia`:

```python
ARCH_MAP = {"9": "hopper", "8": "ampere"}
```

The resolver will look up the current device's arch, use `ARCH_MAP` to
translate it to a subfolder, and load
`_<vendor>/<arch>/ops/` on top of `_<vendor>/ops/`.

#### Step 2.2 — `ops/`

The `ops/` directory holds your vendor-customized operator
implementations. For a custom `add`, drop the implementation in
`ops/add.py`:

```python
# ops/add.py
def add(...):
    ...

__all__ = ["add"]
```

Each op file must define its own `__all__` — the registrar walks each
`*.py` in the `ops/` package and merges every name listed in that
module's `__all__`. Nothing needs to be re-exported from
`ops/__init__.py`; anything not in a module's `__all__` is invisible.

### Step 3 — verify

From a Python shell on your device:

```python
import flaggems_sglang

flaggems_sglang.device                    # -> your device_name
flaggems_sglang.vendor_name               # -> your vendor_name
flaggems_sglang.all_registered_ops()      # -> includes your vendor overrides
flaggems_sglang.get_op("add").__module__  # -> _<vendor>.ops.add or arch path
```

## Multi-level operator routing

`flaggems_sglang` picks the operator implementation for the current
device using three priority levels — later levels override earlier ones
on name collision:

| Priority | Source                                                            | Purpose                     |
|----------|-------------------------------------------------------------------|-----------------------------|
| 0        | `flaggems_sglang.ops`                                             | Generic Triton fallback     |
| 1        | `runtime/backend/_<vendor>/ops`                                   | Per-vendor specialization   |
| 2        | `runtime/backend/_<vendor>/<arch>/ops` (e.g. `_nvidia/hopper/ops`) | Per-arch specialization     |

Routing is driven by function **name**. A vendor override for
an op simply defines `def my_op(...)` in
`_<vendor>/ops/my_op.py` with
`__all__ = ["my_op"]` at the module level — the registrar
discovers it by walking the `ops/` package, no `ops/__init__.py`
re-export required. Missing operators automatically fall back to the
generic `flaggems_sglang.ops.*` implementation, so vendors only ship
the kernels they actually specialize.

The resolver lives in `runtime/op_registrar.py` (`OpRegistrar`) and is
invoked once from `flaggems_sglang/__init__.py`. To introspect the
routing result at runtime:

```python
import flaggems_sglang

flaggems_sglang.all_registered_ops()          # -> ['silu_and_mul', 'mrope_fused', ...]
flaggems_sglang.get_op("silu_and_mul")        # -> resolved callable for current device
flaggems_sglang.silu_and_mul.__module__       # -> module the resolved impl came from
```

## Adding operators to an existing backend

Once your vendor folder is in place, adding or overriding individual
operators follows the standard operator-contribution flow. See
[`docs/CONTRIBUTING.md`](../../../../docs/CONTRIBUTING.md) for:

- Choosing the right tier (generic / vendor / arch).
- File layout, `__all__`, and Apache 2.0 header requirements.
- Adding correctness tests (`tests/test_<op>.py`) and benchmarks
  (`benchmark/test_<op>.py`).
- Code style (pre-commit, black, isort, flake8) and CI expectations.
- The FlagOS × SGLang competition submission flow.
