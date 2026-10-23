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
flaggems_sglang - DNN operations implemented with Triton.

Operators are routed through a three-level registrar
(``runtime.op_registrar.OpRegistrar``):

    generic ops < vendor ops < arch ops

The resolved implementations are attached directly to this package's
namespace, so callers use ``flaggems_sglang.<op>(...)`` regardless of the
underlying vendor.
"""

from typing import Callable, List, Optional

from flaggems_sglang import runtime  # noqa: F401
from flaggems_sglang import testing  # noqa: F401
from flaggems_sglang.runtime.op_registrar import OpRegistrar

device = runtime.device.name
vendor_name = runtime.device.vendor_name

_op_registrar = OpRegistrar()
_op_registrar.apply(globals())


def all_registered_ops() -> List[str]:
    """Return the names of all operators currently resolved on the package."""
    return _op_registrar.names()


def get_op(name: str) -> Optional[Callable]:
    """Look up a resolved operator by name; ``None`` if not registered."""
    return _op_registrar.get(name)


__version__ = "0.1.0"

__all__ = ["all_registered_ops", "get_op", "device", "vendor_name"] + list(
    _op_registrar.names()
)
