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

"""Benchmark base class for the operators implemented in this repository.

``benchmark/performance_utils.py``, ``benchmark/conftest.py`` and
``benchmark/core_shapes.yaml`` are vendored from upstream FlagGems, and that
shape file has no entries for the ops here. The inherited
:meth:`Benchmark.set_shapes` walks the MRO looking for a matching key and
would therefore settle on the generic ``Benchmark:`` shapes from the vendored
YAML. Declaring shapes on the subclass instead keeps them next to the input
builder that consumes them, and leaves the vendored file untouched so it can
be re-synced from upstream.
"""

from . import conftest
from .attri_util import BenchLevel
from .performance_utils import Benchmark


class OpBenchmark(Benchmark):
    """Shape handling shared by this repository's operator benchmarks.

    Subclasses declare ``CORE_SHAPES`` (always benchmarked) and optionally
    ``MORE_SHAPES`` (added by ``--level comprehensive``, the default), plus a
    ``DEFAULT_SHAPE_DESC`` naming the fields of a shape tuple.
    """

    CORE_SHAPES: list = []
    MORE_SHAPES: list = []

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.CORE_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        if self.MORE_SHAPES and _is_comprehensive():
            self.shapes = list(
                dict.fromkeys(self.shapes + list(self.MORE_SHAPES))
            )

    def set_more_shapes(self):
        # Shapes are merged by set_shapes above; returning None keeps the
        # vendored base class from merging them a second time.
        return None


def _is_comprehensive() -> bool:
    # ``conftest.Config`` is rebound by ``pytest_configure``, so read it off
    # the module rather than binding the (initially ``None``) value here.
    config = conftest.Config
    if config is None:
        return False
    return config.bench_level == BenchLevel.COMPREHENSIVE and not config.query


__all__ = ["OpBenchmark"]
