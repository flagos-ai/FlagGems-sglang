# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load standalone reference sources without importing the runtime backend."""

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src/flaggems_sglang/reference"


def load(name):
    path = REFERENCE / (name + ".py")
    spec = importlib.util.spec_from_file_location(
        "_verify_" + name.replace("/", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_source():
    path = REFERENCE / "native/mrotary_embedding.py"
    return dict(
        repository="https://github.com/sgl-project/sglang",
        version="v0.5.11",
        upstream_blob="0a8dc2c33c7bbb2acb7e02b10796d210c5c079fc",
        implementation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
