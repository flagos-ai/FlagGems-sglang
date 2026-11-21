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
Multi-level operator registrar.

Priority (higher wins on name collision):

    Level 0: ``flaggems_sglang.ops.*``                       (generic)
    Level 1: ``flaggems_sglang.runtime.backend._<vendor>.ops`` (vendor)
    Level 2: ``flaggems_sglang.runtime.backend._<vendor>.<arch>.ops`` (arch)

Every tier is discovered the same way: walk the ``ops`` package,
import each ``*.py`` module, and merge every function listed in that
module's ``__all__``. Vendor / arch overrides just define a same-named
function in a file whose ``__all__`` lists it — no ``ops/__init__.py``
re-export required.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from typing import Callable, Dict, List, Optional

from .backend import BackendArchEvent
from .backend.device import DeviceDetector

class OpRegistrar:
    """Collect and route operators from generic / vendor / arch sources."""

    def __init__(self, generic_pkg: str = "flaggems_sglang.ops"):
        self._generic_pkg = generic_pkg
        self._device = DeviceDetector()
        self._resolved: Dict[str, Callable] = {}
        # A serving profile can select compatible modules before imports.
        self._module_allowlist = {
            name.strip()
            for name in os.environ.get("FLAGGEMS_SGLANG_OP_MODULES", "").split(",")
            if name.strip()
        }

    def apply(
        self, target_globals: Optional[Dict] = None
    ) -> Dict[str, Callable]:
        """
        Resolve the operator map and (optionally) inject it into a
        target namespace dict (typically the caller's ``globals()``).
        """
        self._resolved.clear()
        self._collect_generic()
        self._collect_vendor()
        self._collect_arch()
        if target_globals is not None:
            for name, fn in self._resolved.items():
                target_globals[name] = fn
        return dict(self._resolved)

    @property
    def resolved(self) -> Dict[str, Callable]:
        return dict(self._resolved)

    def names(self) -> List[str]:
        return sorted(self._resolved.keys())

    def get(self, name: str) -> Optional[Callable]:
        return self._resolved.get(name)

    def _collect_generic(self) -> None:
        self._collect_from_package(self._generic_pkg)

    def _collect_vendor(self) -> None:
        vendor = self._device.vendor_name
        if not vendor:
            return
        self._collect_from_package(
            f"flaggems_sglang.runtime.backend._{vendor}.ops"
        )

    def _collect_arch(self) -> None:
        event = BackendArchEvent()
        if not event.has_arch or not event.arch:
            return
        vendor = self._device.vendor_name
        self._collect_from_package(
            f"flaggems_sglang.runtime.backend._{vendor}.{event.arch}.ops"
        )

    def _collect_from_package(self, pkg_name: str) -> None:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            return
        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return
        for mod_info in pkgutil.iter_modules(pkg_path):
            if mod_info.ispkg or mod_info.name.startswith("_"):
                continue
            if self._module_allowlist and mod_info.name not in self._module_allowlist:
                continue
            mod = importlib.import_module(f"{pkg_name}.{mod_info.name}")
            self._merge_from_module(mod)

    def _merge_from_module(self, module) -> None:
        exported = getattr(module, "__all__", None)
        if exported is None:
            return
        for name in exported:
            fn = getattr(module, name, None)
            if fn is None or not inspect.isfunction(fn):
                continue
            self._resolved[name] = fn


__all__ = ["OpRegistrar"]
