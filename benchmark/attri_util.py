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

import itertools
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple

import torch

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

# (tokens, heads, head_dim, transposed_gate) for fused attention gating.
FUSED_SIGMOID_MUL_SHAPES = [
    (tokens, 32, 128, transposed)
    for tokens in (1, 17, 256)
    for transposed in (0, 1)
]

INT_DTYPES = [torch.int16, torch.int32]
BOOL_DTYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex64]

DEFAULT_WARMUP_COUNT = 1000
DEFAULT_ITER_COUNT = 100

# LEGACY_SHAPES are maintained for legacy benchmark SIZE
# settings and may be removed in the future.
# Do not reference this elsewhere.
LEGACY_SHAPES = [i * 64 for i in range(1, 22, 5)]
LEGACY_NON_DNN_SHAPES = [(1024, shape) for shape in LEGACY_SHAPES]
LEGACY_DNN_SHAPES = [(16, shape, shape, shape) for shape in LEGACY_SHAPES]

# Default shapes settings
DEFAULT_SHAPES = [
    (1024 * 1024 * 1024,),  # from perf
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),  # from perf
]

# (B, H, HV, K, V, pool_size) tuples for fused recurrent decode benchmarks.
# Shapes match Qwen3.5-35B-A3B decode serving configurations.
FUSED_RECURRENT_BENCH_SHAPES = [
    (1, 8, 16, 128, 128, 32),
    (4, 8, 16, 128, 128, 32),
    (8, 8, 16, 128, 128, 64),
    (16, 8, 16, 128, 128, 64),
    (32, 8, 16, 128, 128, 128),
    (64, 8, 16, 128, 128, 128),
    (128, 8, 16, 128, 128, 256),
    (256, 8, 16, 128, 128, 512),
]

# (M, K, N, E, topk) tuples for FusedMoE benchmark.
# Covers decode (small M) and prefill (large M) scenarios with
# typical MoE dimensions from Mixtral/DeepSeek-class models.
FUSED_MOE_BENCH_SHAPES = [
    (1, 1280, 3072, 64, 8),
    (4, 1280, 3072, 64, 8),
    (17, 1280, 3072, 64, 8),
    (64, 1280, 3072, 64, 8),
    (1024, 1280, 3072, 64, 8),
    (2048, 1280, 3072, 64, 8),
    (4096, 1280, 3072, 64, 8),
]

# (M, N) tuples for GemmaRMSNorm benchmark.  N values match Gemma model
# hidden dimensions: 512 (2B), 1024, 2048 (7B), 3072, 4096, 5120, 8192.
# Small batch sizes representative of decode serving scenarios.
GEMMA_RMS_NORM_BENCH_SHAPES = [
    (1, 512),
    (4, 512),
    (1, 1024),
    (4, 1024),
    (1, 2048),
    (4, 2048),
    (1, 3072),
    (4, 3072),
    (1, 4096),
    (4, 4096),
    (1, 5120),
    (4, 5120),
    (1, 8192),
    (4, 8192),
]


def model_shapes():
    # batch sizes * seq lengths
    BS = [1, 2, 3, 4, 8, 98, 256, 8192]
    # attn: wqkv, wo; ffn: w13, w2
    NK = [
        # extract from llama3-8b
        (1024, 4096),
        (128256, 4096),
        (14336, 4096),
        (4096, 14336),
        (4096, 4096),
        (6144, 4096),
        (28672, 4096),
        # extract from qwen2.5-7b
        (3584, 3584),
        (18944, 3584),
        (3584, 18944),
        (152064, 3584),
        (37888, 3584),
        (512, 3584),
        (4608, 3584),
    ]

    return [(4, bs, n, k) for bs, (n, k) in itertools.product(BS, NK)]


@dataclass
class BenchmarkMetrics:
    # Stable workload identity for case listing/replay (None in legacy runs).
    case_id: Optional[str] = None
    # Legacy shape information for backward compatibility
    # This field corresponds to the 'size' field in
    # the previous version's benchmark.
    legacy_shape: Optional[int] = None
    # Detailed size info
    shape_detail: Optional[Tuple[int, ...]] = None
    # Latency base in ms
    latency_base: Optional[float] = None
    # Latency in ms
    latency: Optional[float] = None
    gbps_base: Optional[float] = None
    gbps: Optional[float] = None
    # Speedup over baseline
    speedup: Optional[float] = None
    # Accuracy over baseline (not implemented yet)
    accuracy: Optional[float] = None
    # TFLOPS (not implemented yet)
    tflops: Optional[float] = None
    # Utilization (not implemented yet)
    utilization: Optional[float] = None
    # Speedup compared to base data
    compared_speedup: Optional[float] = None
    # Which implementation was timed as the candidate ("override" when a
    # dynamically injected candidate was invoked); None for the default path.
    candidate_source: Optional[str] = None
    # Why the baseline comparison was skipped (op/shape/dtype missing).
    # None means "matched" or "--base-data not enabled". Excluded from metrics.
    base_miss_reason: Optional[str] = None
    # Error message
    error_msg: Optional[str] = None


ALL_AVAILABLE_METRICS = set(
    map(lambda x: x.name, fields(BenchmarkMetrics))
) - {
    "case_id",
    "legacy_shape",
    "shape_detail",
    "candidate_source",
    "base_miss_reason",
}

DEFAULT_METRICS = [
    metric
    for metric in ["latency_base", "latency", "speedup"]
    if metric in ALL_AVAILABLE_METRICS
]


def check_metric_dependencies(
    requested_metrics: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Checks if the requested metrics satisfy their dependencies.
    Returns True if the dependencies are satisfied, otherwise False.
    """
    # Predefined dependencies between metrics
    buildin_dependencies = {
        "speedup": ["latency", "latency_base"],
        "utilization": ["latency", "tflops"],
    }
    unsatisfied_metrics: list = []
    if requested_metrics is None:
        return unsatisfied_metrics

    satisfied_metrics = set()
    for metric in requested_metrics:
        if metric not in buildin_dependencies:
            # If the metric has no dependencies, it's automatically satisfied
            satisfied_metrics.add(metric)
        else:
            required_metrics = buildin_dependencies[metric]
            # Check if all dependencies are in the satisfied metrics list
            if not all(req in satisfied_metrics for req in required_metrics):
                unsatisfied_metrics.append(metric)
            else:
                satisfied_metrics.add(metric)
    return unsatisfied_metrics


def get_recommended_shapes(
    op_name: str, op_specified_shapes: Optional[List[Tuple[int, ...]]]
):
    def _shapes_sort(shapes):
        shapes = [
            shape if isinstance(shape, tuple) else (shape,) for shape in shapes
        ]
        return sorted(shapes, key=lambda x: torch.tensor(x).prod().item())

    if op_specified_shapes:
        # TODO: handle situation that list as the basic element in shape.
        return _shapes_sort(op_specified_shapes)
    return _shapes_sort(DEFAULT_SHAPES)


class BenchMode(Enum):
    KERNEL = "kernel"
    OPERATOR = "operator"
    WRAPPER = "wrapper"


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


@dataclass
class OperationAttribute:
    op_name: str
    # Recommended core benchmark shapes for the given operation
    recommended_core_shapes: List[Tuple[int, ...]]
    shape_desc: str

    def __str__(self) -> str:
        shapes_label = "Recommended Core Shapes[" + self.shape_desc + "]"
        return (
            f"{'Operator name':<40}"
            f" |  {self.op_name}\n"
            f"{shapes_label:<40}"
            f" |  {self.recommended_core_shapes}\n"
        )

    def to_dict(self) -> dict:
        return self.__dict__


def custom_json_encoder(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable"
    )


# ---- NVIDIA baseline lookup ("vs Base" column) ------------------------------
# Used by the benchmark run loop (performance_utils.py) to compare each result
# against a baseline latency loaded from --base-data. Lives in attri_util.py
# (re-exported by consts.py) because this is the lowest layer both benchmark
# and tooling import, so there is no import cycle.

# torch dtype string -> scaling-factor dtype key (see _scaling_factors block).
_DTYPE_TO_SF_KEY = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.int8": "int8",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e5m2": "fp8",
}

# baseline bottleneck unit -> scaling-factor resource group. mem is
# dtype-independent (memory bandwidth); tensor/cuda pick a per-dtype compute
# factor (Tensor Core vs CUDA Core/vector).
_BOTTLENECK_TO_SF = {
    "mem": "bandwidth_gbps",
    "tensor": "tensor",
    "cuda": "vector",
}


def _first_tensor_shape(shape_detail):
    """Return the first tensor shape found in a record_shapes() result as a
    plain list of ints, or None if there is no tensor shape.

    record_shapes() encodes tensors as torch.Size and may nest them inside
    lists/tuples/dicts (and wrap args+kwargs in a top-level tuple), so walk the
    structure depth-first and return the first torch.Size we hit.
    """
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            return list(item)
        if isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return None


def _all_tensor_shapes_from_record(shape_detail):
    """Collect **all** tensor shapes from a record_shapes() result, returned as
    a sorted tuple.

    Unlike _first_tensor_shape this does not stop at the first tensor; it walks
    the whole (args+kwargs nested list/tuple/dict) structure and gathers every
    torch.Size. The multiset of tensor shapes almost always uniquely identifies
    a collection point within one op, so using it as a signature works around
    the "baseline key_shape is a semantic composite key that does not match the
    first tensor shape" matching failures (see the signature fallback in
    lookup_base_record_ex).
    """
    shapes = []
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            shapes.append(tuple(item))
        elif isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return tuple(sorted(shapes))


def _all_scalars_from_record(shape_detail):
    """Collect scalars (int/float/str/bool) from a record_shapes() result,
    returned sorted.

    Only used for the few ops distinguished by a scalar rather than tensor
    shape. torch.Size is handled separately above, so here only loose scalars
    are gathered. bool must be checked before int (isinstance(True, int)).
    """
    scalars = []
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            continue
        if isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
        elif isinstance(item, bool):
            scalars.append(repr(item))
        elif isinstance(item, (int, float, str)):
            scalars.append(repr(item))
    return tuple(sorted(scalars))


def _base_shapes_from_config(cfg):
    """Collect all tensor shapes from a baseline record's config.inputs,
    returned as a sorted tuple.

    Symmetric with _all_tensor_shapes_from_record: the collector writes each
    input tensor as {"shape": [...]} and scalars as {"scalar": v}. Returns None
    when config/inputs is missing, meaning that record cannot be signature
    matched (only its raw key can).
    """
    if not isinstance(cfg, dict):
        return None
    inputs = cfg.get("inputs")
    if not isinstance(inputs, dict):
        return None
    shapes = []
    for v in inputs.values():
        if isinstance(v, dict) and isinstance(v.get("shape"), list):
            shapes.append(tuple(v["shape"]))
    return tuple(sorted(shapes))


def _base_scalars_from_config(cfg):
    """Collect scalar values from a baseline config.inputs, returned sorted;
    symmetric with the record side."""
    scalars = []
    inputs = (cfg or {}).get("inputs", {})
    if isinstance(inputs, dict):
        for v in inputs.values():
            if isinstance(v, dict) and "scalar" in v:
                scalars.append(repr(v["scalar"]))
    return tuple(sorted(scalars))


# Per (base_data, op_name) "tensor signature -> shape key" index, cached by the
# id of base_data to avoid rebuilding for every collection point. Values are
# {dtype: {tensor_sig: [shape_key, ...]}}.
_SIG_INDEX_CACHE = {}


def _build_signature_index(base_data, op_name):
    """Build the {dtype: {tensor_sig: [shape_key, ...]}} index for one op
    (cached)."""
    cache_key = (id(base_data), op_name)
    cached = _SIG_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = {}
    op_entry = base_data.get(op_name) or {}
    for shape_key, per_dtype in op_entry.get("shapes", {}).items():
        if not isinstance(per_dtype, dict):
            continue
        for dtype_str, rec in per_dtype.items():
            if not isinstance(rec, dict):
                continue
            sig = _base_shapes_from_config(rec.get("config"))
            if not sig:
                continue
            index.setdefault(dtype_str, {}).setdefault(sig, []).append(
                shape_key
            )
    _SIG_INDEX_CACHE[cache_key] = index
    return index


def _match_by_signature(base_data, op_name, dtype, shape_detail):
    """Tensor-signature fallback match: returns (record, matched_shape_key) or
    (None, None).

    Called when the direct str(first_shape) match fails. Looks up by the
    multiset of all tensor shapes under this (op, dtype); if several shape keys
    share the same tensor signature (a few ops), disambiguates with the scalar
    set. If it still cannot be uniquely located, gives up (returns None) and the
    caller reports a miss by the normal path, never guessing.
    """
    index = _build_signature_index(base_data, op_name)
    per_dtype = index.get(str(dtype))
    if not per_dtype:
        return None, None
    sig = _all_tensor_shapes_from_record(shape_detail)
    keys = per_dtype.get(sig)
    if not keys:
        return None, None
    op_shapes = base_data[op_name]["shapes"]
    if len(keys) == 1:
        return op_shapes[keys[0]].get(str(dtype)), keys[0]
    # Tensor-signature collision: disambiguate with the scalar set.
    rec_scalars = _all_scalars_from_record(shape_detail)
    for k in keys:
        rec = op_shapes[k].get(str(dtype))
        if rec is None:
            continue
        base_scalars = _base_scalars_from_config(rec.get("config"))
        # Baseline scalars should be a subset of the record-side scalars (the
        # record side often carries extra loose ints such as strides).
        if base_scalars and set(base_scalars) <= set(rec_scalars):
            return rec, k
    return None, None


def lookup_base_record_ex(base_data, op_name, dtype, shape_detail):
    """Look up the NVIDIA baseline record, three levels deep:
    op_name -> shape -> dtype. Returns a (record, miss_reason) tuple:

    - (record_dict, None)  when every level matched;
    - (None, reason_str)   when any level is missing, where reason_str names
                           exactly which level failed so callers can print a
                           precise diagnostic and persist it.

    Never raises: an unmatched entry just yields a reason. Baseline layout:
        base_data[op_name]["shapes"][str([d0, d1, ...])][str(dtype)]
    The shape key is the string of the first tensor's dimensions for simple
    ops; complex ops fall back to the tensor-signature match.
    """
    try:
        op_entry = base_data.get(op_name)
        if not op_entry:
            return None, f"baseline 中没有算子 '{op_name}'"
        shapes = op_entry.get("shapes", {})
        if isinstance(shape_detail, dict) and "baseline_key" in shape_detail:
            key = shape_detail["baseline_key"]
            rec = shapes.get(key, {}).get(str(dtype))
            if rec is None:
                return (
                    None,
                    f"baseline 没有精确匹配 '{op_name}' key={key} dtype={dtype}",
                )
            return rec, None
        first_shape = _first_tensor_shape(shape_detail)
        if first_shape is None:
            return None, (
                f"算子 '{op_name}' 的 shape 无法表达为张量维度 "
                f"(shape_detail={shape_detail})"
            )
        # Prefer the raw key (str(first tensor shape)) directly -- for simple
        # ops the key_shape is exactly the first tensor shape, so this hits
        # often and without ambiguity.
        per_dtype = shapes.get(str(first_shape))
        if per_dtype:
            rec = per_dtype.get(str(dtype))
            if rec is not None:
                return rec, None
        # Direct lookup missed: most complex ops have a semantic composite or
        # string key_shape that differs from the first tensor shape. Fall back
        # to the "multiset of all tensor shapes" signature match -- symmetric
        # with the collector's config.inputs, so both sides agree.
        rec, matched_key = _match_by_signature(
            base_data, op_name, dtype, shape_detail
        )
        if rec is not None:
            return rec, None
        # Still missed: distinguish "this shape signature is absent from the
        # baseline" from "shape present but dtype missing" for a precise
        # diagnostic.
        if (
            str(first_shape) in shapes
            and shapes[str(first_shape)].get(str(dtype)) is None
        ):
            return None, (
                f"算子 '{op_name}' shape {first_shape} 在 baseline 中没有 "
                f"对应的 dtype {dtype}"
            )
        return None, (
            f"算子 '{op_name}' 在 baseline 中没有匹配的 shape"
            f"（第一个张量={first_shape}，已尝试张量签名回退）"
        )
    except (AttributeError, TypeError) as e:
        return None, f"baseline 查找 '{op_name}' 时出错: {e}"


def lookup_base_record(base_data, op_name, dtype, shape_detail):
    """Backward-compatible wrapper: returns just the record dict, or None.

    Prefer lookup_base_record_ex when the miss reason is needed.
    """
    rec, _ = lookup_base_record_ex(base_data, op_name, dtype, shape_detail)
    return rec


def _detect_self_chip():
    """Best-effort name of the chip this benchmark runs on, e.g. "NVIDIA H800".

    Used to pick the right scaling factor when one vendor has several chips
    (H800 vs H100) and to recognize when we're running on the baseline's own
    reference chip. Returns "" when the device name can't be read; callers then
    fall back to vendor-only matching.
    """
    try:
        return torch.cuda.get_device_name()
    except Exception:
        return ""


# Cache the device name once; it doesn't change within a run.
_SELF_CHIP = _detect_self_chip()


def _chip_matches(chip_name, device_name):
    """True if a spec's chip name (e.g. "H800") identifies this device
    (e.g. "NVIDIA H800 80GB"). Case-insensitive substring match, which
    tolerates the vendor prefix / memory suffix get_device_name() adds."""
    if not chip_name or not device_name:
        return False
    return chip_name.lower() in device_name.lower()


def lookup_scaling_factor(
    base_data, vendor, bottle_neck_unit, dtype, self_chip=None
):
    """Return this chip's hardware scaling factor (chip_peak / reference_peak)
    for the resource that bottlenecks the baseline record, or None.

    Factor comes from base_data["_scaling_factors"]. Matching is by vendor AND
    concrete chip model, so that e.g. nvidia H800 and H100 don't collide.
    Prefers measured over nominal. Returns None when the vendor / chip /
    resource / dtype isn't covered, so the caller falls back to the raw latency
    ratio.

    The reference chip is itself listed under "chips" with all factors equal to
    1.0, so it matches by model like any other chip.
    """
    try:
        sf = base_data.get("_scaling_factors")
        resource = _BOTTLENECK_TO_SF.get(bottle_neck_unit)
        if not sf or resource is None:
            return None
        if self_chip is None:
            self_chip = _SELF_CHIP
        candidates = [
            c for c in sf.get("chips", []) if c.get("vendor") == vendor
        ]
        if not candidates:
            return None
        # Prefer the candidate whose chip model matches this device; only when a
        # single vendor chip exists do we accept it without a model match.
        chip = next(
            (c for c in candidates if _chip_matches(c.get("chip"), self_chip)),
            None,
        )
        if chip is None:
            chip = candidates[0] if len(candidates) == 1 else None
        if chip is None:
            return None
        for prefer in ("measured", "nominal"):
            block = chip.get(prefer)
            if not block:
                continue
            if resource == "bandwidth_gbps":
                val = block.get("bandwidth_gbps")
            else:
                val = block.get(resource, {}).get(
                    _DTYPE_TO_SF_KEY.get(str(dtype))
                )
            if val:
                return val
        return None
    except (AttributeError, TypeError):
        return None


@dataclass
class BenchmarkResult:
    """Record the benchmark result for each operator."""

    # Unique name of the operator
    op_name: str
    dtype: str
    mode: str
    level: str
    # Benchmark results
    result: List[BenchmarkMetrics]

    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance"
            f" Test (dtype={self.dtype},"
            f" mode={self.mode},"
            f" level={self.level})\n"
        )
        col_names = [
            f"{'Status':<10}",
            f"{'Torch Latency (ms)':>20}",
            f"{'Gems Latency (ms)':>20}",
            f"{'Gems Speedup':>20}",
        ]
        # Only surface the baseline column when baseline data was actually
        # matched for this op; otherwise the output stays identical to before.
        show_vs_base = any(m.compared_speedup is not None for m in self.result)
        if show_vs_base:
            col_names.append(f"{'vs Base':>20}")
        if self.result[0].tflops and self.result[0].tflops != 0.0:
            col_names.append(f"{'TFLOPS':>20}")
        if self.result[0].gbps is not None:
            col_names.append(f"{'Torch GBPS ':>20}")
            col_names.append(f"{'Gems GBPS ':>20}")
        col_names.append(f"{'Size Detail':>20}\n")
        header_col_names = " ".join(col_names)
        header_break = "-" * len(header_col_names) + "\n"
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(
            self._format_metrics(ele, show_vs_base) for ele in self.result
        )
        return header + metrics_lines

    def _format_metrics(
        self, metrics: BenchmarkMetrics, show_vs_base: bool = False
    ) -> str:
        # self.gen_legacy_shape(metrics)
        # legacy_shape_str = (
        #     metrics.legacy_shape
        #     if metrics.legacy_shape is not None
        #     else "N/A"
        # )
        latency_base_str = (
            f"{metrics.latency_base:.6f}"
            if metrics.latency_base is not None
            else "N/A"
        )
        latency_str = (
            f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        )
        speedup_str = (
            f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        )
        compared_speedup_str = (
            f"{metrics.compared_speedup:.3f}"
            if metrics.compared_speedup is not None
            else "N/A"
        )
        torch_gbps_str = (
            f"{metrics.gbps_base:.3f}"
            if metrics.gbps_base is not None
            else "N/A"
        )
        gems_gbps_str = (
            f"{metrics.gbps:.3f}" if metrics.gbps is not None else "N/A"
        )
        if metrics.tflops and metrics.tflops != 0.0:
            tflops_str = (
                f"{metrics.tflops:.3f}"
                if metrics.tflops is not None
                else "N/A"
            )
        shape_detail_str = (
            metrics.shape_detail if metrics.shape_detail is not None else "N/A"
        )
        status = "SUCCESS" if metrics.error_msg is None else "FAILED"
        data_line = (
            f"{status:<10}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
        )
        if show_vs_base:
            data_line += f"{compared_speedup_str:>20}"
        if metrics.tflops and metrics.tflops != 0.0:
            data_line += f"{tflops_str:>20}"
        if metrics.gbps is not None:
            data_line += f"{torch_gbps_str:>20}{gems_gbps_str:>20}"
        data_line += " " * 10
        data_line += f"{shape_detail_str}\n"
        return data_line

    def gen_legacy_shape(self, metrics: BenchmarkMetrics) -> Optional[int]:
        first_shape = (
            metrics.shape_detail[0]
            if isinstance(metrics.shape_detail, list)
            else None
        )
        to_record_shape = (
            tuple(first_shape) if isinstance(first_shape, torch.Size) else None
        )

        if to_record_shape in LEGACY_NON_DNN_SHAPES:
            metrics.legacy_shape = to_record_shape[-1]
        elif (
            isinstance(to_record_shape, tuple)
            and len(to_record_shape) == 2
            and to_record_shape[0] == 1024
        ):
            metrics.legacy_shape = to_record_shape[-1]
        else:
            metrics.legacy_shape = None
        return None

    def to_json(self) -> str:
        import json

        # Convert to dict and handle tuple serialization for shape_detail
        result_dict = asdict(self)
        return json.dumps(result_dict, default=custom_json_encoder)

    def to_dict(self) -> dict:
        return self.__dict__
