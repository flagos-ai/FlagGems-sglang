# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Both libraries must coexist without sharing bare backend-module caches."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "first,second",
    [("flaggems_sglang", "flag_gems"), ("flag_gems", "flaggems_sglang")],
)
def test_both_import_orders_keep_vendor_and_arch_namespaces(first, second):
    code = f"""
import {first}
import {second}
from flaggems_sglang.runtime import backend as s
from flag_gems.runtime import backend as g
assert s.get_vendor_module('nvidia').__name__ == 'flaggems_sglang.runtime.backend._nvidia'
assert g.get_vendor_module('nvidia').__file__.startswith(__import__('flag_gems').__path__[0])
assert s.get_vendor_info().__class__.__module__.startswith('flaggems_sglang.')
assert __import__('sys').modules[g.get_vendor_info().__class__.__module__].__file__.startswith(__import__('flag_gems').__path__[0])
assert s.BackendArchEvent().arch_module.__name__.startswith('flaggems_sglang.')
print('BOTH_IMPORT_ORDERS_PASS')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
