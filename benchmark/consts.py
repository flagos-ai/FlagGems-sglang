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

"""Compatibility alias for :mod:`benchmark.attri_util`.

Upstream FlagGems renamed ``benchmark/attri_util.py`` to
``benchmark/consts.py``; ``benchmark/conftest.py`` and ``tools/run_tests.py``
are vendored from upstream and import ``consts``. This repository still uses
the ``attri_util`` name, so re-export it here instead of forking the vendored
files. Names upstream added after the split are defined below.
"""

import torch

from .attri_util import *  # noqa: F401,F403
from .attri_util import (  # noqa: F401
    ALL_AVAILABLE_METRICS,
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_ITER_COUNT,
    DEFAULT_METRICS,
    DEFAULT_SHAPES,
    DEFAULT_WARMUP_COUNT,
    FLOAT_DTYPES,
    INT_DTYPES,
    LEGACY_SHAPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchMode,
    OperationAttribute,
    check_metric_dependencies,
    custom_json_encoder,
    get_recommended_shapes,
    model_shapes,
)

# Upstream spellings of the warmup/iteration defaults.
DEFAULT_WARMUP_TIME = DEFAULT_WARMUP_COUNT
DEFAULT_ITER_TIME = DEFAULT_ITER_COUNT

# Upstream spellings of the legacy shape lists.
from .attri_util import (  # noqa: E402,F401
    LEGACY_DNN_SHAPES as LEGACY_BLAS_SHAPES,
)
from .attri_util import (  # noqa: E402,F401
    LEGACY_NON_DNN_SHAPES as LEGACY_NON_BLAS_SHAPES,
)

EXTRA_INT_DTYPES = [torch.int8, torch.uint8, torch.int64]

# Mapping used by tools/run_tests.py to shorten dtype names in reports.
DTYPE_MAP = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.int64": "int64",
    "torch.bool": "bool",
    "torch.complex64": "cf64",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
}
