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

"""Shared benchmark entry point for this repository's operators.

Each op benchmark reuses upstream's :class:`GenericBenchmark`: the op supplies
an ``input_fn(shape, cur_dtype, device)`` generator plus its shape list, and
the base class handles dtype/metric selection, warmup, timing against the
reference and the ``Operator: ... Performance Test`` report. Ops therefore
need no benchmark subclass of their own -- see ``test_silu_and_mul.py``.

Shapes are constructor arguments rather than YAML entries. ``core_shapes.yaml``
is vendored from upstream FlagGems and has no keys for this repository's ops,
so the inherited ``set_shapes`` would walk the MRO and silently fall back to
the generic ``Benchmark:`` shapes. Keeping shapes out of that file leaves it
re-syncable.
"""

from typing import Any, Iterable, Optional, Sequence

from . import conftest
from .attri_util import BenchLevel
from .performance_utils import GenericBenchmark


class OpBenchmark(GenericBenchmark):
    """:class:`GenericBenchmark` with shapes supplied by the caller.

    ``shapes`` always runs; ``more_shapes`` is appended at the default
    ``--level comprehensive`` and skipped for ``--level core``.
    """

    def __init__(
        self,
        *args: Any,
        shapes: Iterable[Sequence[int]],
        shape_desc: str,
        more_shapes: Optional[Iterable[Sequence[int]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.core_shapes = [tuple(shape) for shape in shapes]
        self.more_shapes = [tuple(shape) for shape in more_shapes or ()]
        self.shapes = list(self.core_shapes)
        self.shape_desc = shape_desc

    def set_shapes(self, shape_file_path: Optional[str] = None) -> None:
        self.shapes = list(self.core_shapes)
        if self.more_shapes and _is_comprehensive():
            self.shapes = list(dict.fromkeys(self.shapes + self.more_shapes))

    def set_more_shapes(self) -> None:
        # Shapes are merged by set_shapes above. The generic 1D/2D/3D shapes
        # GenericBenchmark would add here don't fit these operators.
        return None


def _is_comprehensive() -> bool:
    # ``conftest.Config`` is rebound by ``pytest_configure``, so read it off
    # the module rather than binding the (initially ``None``) value here.
    config = conftest.Config
    if config is None:
        return False
    return config.bench_level == BenchLevel.COMPREHENSIVE and not config.query


__all__ = ["OpBenchmark"]
