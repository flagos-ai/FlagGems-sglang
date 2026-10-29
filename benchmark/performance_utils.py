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

import gc
import importlib
import json
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Collection, Generator, List, Optional, Tuple

import pytest
import torch
import triton
import yaml  # type: ignore[import-untyped]

import flaggems_sglang
from flaggems_sglang.reference_candidate import (
    ReferenceContractError,
    reference_execution,
    reference_mode_enabled,
    select_candidate,
)

from .attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    DEFAULT_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchMode,
    OperationAttribute,
    check_metric_dependencies,
    lookup_base_record_ex,
    lookup_scaling_factor,
)
from .cases import BenchmarkCaseList, BenchmarkCasePlan, BenchmarkCaseSpec
from .conftest import (
    Config,
    emit_record_logger,
    update_case_list,
    update_result,
)
from .reference import reference_failure

torch_backend_device = flaggems_sglang.runtime.torch_backend_device
torch_device_fn = flaggems_sglang.runtime.torch_device_fn
device = flaggems_sglang.device
vendor_name = flaggems_sglang.vendor_name
if device == "musa":
    torch.backends.mudnn.allow_tf32 = False
elif device == "npu":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    # Attempt to disallow tf32. Backends such as gcu have no
    # torch.backends.<device> module, leaving torch_backend_device as None.
    try:
        torch_backend_device.matmul.allow_tf32 = False
    except Exception:
        pass

ELEMENTWISE_PERF_SHAPES = [
    # Launch overhead, non-power-of-two, and 1D throughput coverage.
    (1,),
    (16,),
    (127,),
    (1024,),
    (65536,),
    (1024 * 1024,),
    (16 * 1024 * 1024,),
    # Non-aligned 2D and matrix-like layouts.
    (17, 31),
    (1024, 1025),
    (4096, 4096),
    # NLP / LLM activation layouts.
    (1, 2048, 4096),
    (8, 128, 12288),
    # CV activation layouts.
    (1, 3, 224, 224),
    (8, 64, 56, 56),
    (32, 256, 14, 14),
    # 5D layout to catch flattening/indexing assumptions.
    (1, 8, 16, 32, 32),
]


def SkipVersion(module_name, skip_pattern):
    if importlib.util.find_spec(module_name) is None:
        return True
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">"), f"Invalid comparison operator: {cmp}"
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from skip_pattern.")

    try:
        version = importlib.metadata.version(module_name)
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


def do_bench_latency(
    fn: Callable[[], Any],
    warmup: int,
    rep: int,
    grad_to_none: Optional[List[Any]] = None,
) -> float:
    """Return the median latency of ``fn`` in milliseconds.

    ``triton.testing.do_bench`` is not available on every backend:
    triton_ascend ships ``triton.backends.ascend.testing.do_bench_npu``
    instead, and that entry point takes an iteration count rather than
    warmup/repetition durations. Dispatching here means benchmarks stay
    vendor neutral.

    ``warmup``/``rep``/``grad_to_none`` follow ``triton.testing.do_bench`` and
    are ignored on backends whose ``do_bench`` does not accept them.
    """
    if vendor_name == "ascend":
        from triton.backends.ascend.testing import do_bench_npu

        # do_bench_npu requires iterations, rather than duration, so the
        # warmup/repetition durations above do not apply here.
        return do_bench_npu(fn)

    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="median",
        grad_to_none=grad_to_none,
    )


class Benchmark:
    device: str = device
    DEFAULT_METRICS = DEFAULT_METRICS
    DEFAULT_DTYPES = FLOAT_DTYPES
    DEFAULT_SHAPES = DEFAULT_SHAPES
    DEFAULT_SHAPE_DESC = "M, N"
    DEFAULT_SHAPE_FILES = "core_shapes.yaml"
    """
    the base class for the operations benchmark
    """

    def __init__(
        self,
        op_name,
        torch_op,
        dtypes=None,
        is_backward=False,
        is_inplace=False,
        **kwargs,
    ):
        self.op_name = op_name
        if is_backward and self.op_name.find("_backward") == -1:
            self.op_name += "_backward"
        if (
            getattr(Config, "reference_only", False)
            and type(self).run is not Benchmark.run
        ):
            Config.reference_records.append(
                {
                    "nodeid": Config.current_nodeid,
                    "operator": self.op_name,
                    "status": "UNSUPPORTED",
                    "reason": (
                        "custom benchmark run requires an explicit reference "
                        "runner"
                    ),
                }
            )
            pytest.skip(
                "reference-only is unsupported for custom benchmark run"
            )
        self.torch_op = torch_op
        self.gems_op = None
        self.is_backward = is_backward
        self._validate_reference_candidate()
        self.is_inplace = is_inplace
        self._input_iter = None

        # Theoretical supported dtypes, metrics for the operation.
        # These are set by default.
        self.dtypes = dtypes if dtypes is not None else self.DEFAULT_DTYPES
        self.metrics = self.DEFAULT_METRICS
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.shape_file = self.DEFAULT_SHAPE_FILES

        # Actual dtypes and metrics to be used in the benchmark,
        # can be influenced by user input.
        self.to_bench_dtypes = self.dtypes
        self.to_bench_metrics = self.metrics

        # additional properties
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])

    def set_metrics(self, user_desired_metrics: Optional[List[str]]):
        # Validate user-specified metrics
        if user_desired_metrics:
            invalid_metrics = [
                metric
                for metric in user_desired_metrics
                if metric not in self.metrics
            ]
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics: "
                    f"{', '.join(invalid_metrics)}"
                    f" for operation: "
                    f"'{self.op_name}'"
                )
            unsatisfied_metrics = check_metric_dependencies(
                user_desired_metrics
            )
            if unsatisfied_metrics:
                raise ValueError(
                    "Unsatisfied metric dependencies"
                    f": {', '.join(unsatisfied_metrics)}"
                )

        self.to_bench_metrics = user_desired_metrics or self.metrics
        if (
            hasattr(self, "set_more_metrics")
            and callable(getattr(self, "set_more_metrics"))
            and Config.bench_level == BenchLevel.COMPREHENSIVE
            and not Config.query
        ):
            for metric in self.set_more_metrics():
                if metric not in self.to_bench_metrics:
                    self.to_bench_metrics.append(metric)

    def set_more_metrics(self):
        """Base method (optional to override).

        Returns additional shapes if applicable.
        """
        return []

    def set_dtypes(self, user_desired_dtypes: Optional[List[torch.dtype]]):
        # Validate user-specified dtypes
        if user_desired_dtypes and not all(
            dtype in self.dtypes for dtype in user_desired_dtypes
        ):
            invalid_dtypes = [
                dtype
                for dtype in user_desired_dtypes
                if dtype not in self.dtypes
            ]
            raise ValueError(
                "Given dtype(s) '"
                f"{', '.join(str(dtype) for dtype in invalid_dtypes)}'"
                f" can't be supported by"
                f" this op '{self.op_name}'"
            )
        self.to_bench_dtypes = (
            user_desired_dtypes if user_desired_dtypes else self.dtypes
        )

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Validate user-spicified shapes files
        import os

        if not os.path.isfile(shape_file_path):  # type: ignore[arg-type]
            raise FileNotFoundError(
                f"Shape file '{shape_file_path}' does not exist."
            )
        try:
            with open(  # type: ignore[call-overload]
                shape_file_path, "r"
            ) as file:
                yaml_config = yaml.safe_load(file)
                if self.op_name in yaml_config:
                    self.shapes = yaml_config[self.op_name].get(
                        "shapes", self.DEFAULT_SHAPES
                    )
                    self.shape_desc = yaml_config[self.op_name].get(
                        "shape_desc", self.DEFAULT_SHAPE_DESC
                    )
                else:
                    for cls in type(self).__mro__:
                        class_name = cls.__name__
                        if class_name in yaml_config:
                            self.shapes = yaml_config[class_name].get(
                                "shapes", self.DEFAULT_SHAPES
                            )
                            self.shape_desc = yaml_config[class_name].get(
                                "shape_desc", self.DEFAULT_SHAPE_DESC
                            )
                            break
                    else:
                        self.shapes = self.DEFAULT_SHAPES

            self.shapes = [tuple(shape) for shape in self.shapes]
            if vendor_name == "kunlunxin":
                if self.op_name in ["isin", "nonzero"]:
                    # isin oom  # nonzero oot
                    import math

                    self.shapes = [
                        shape
                        for shape in self.shapes
                        if math.prod(shape) < 1024 * 1024
                    ]

            # merge shapes from subclass; if subclass has
            # `set_more_shapes`, call it to merge shapes
            if (
                hasattr(self, "set_more_shapes")
                and callable(getattr(self, "set_more_shapes"))
                and Config.bench_level == BenchLevel.COMPREHENSIVE
                and not Config.query
            ):
                # Merge shapes using subclass-specific logic
                additional_shapes = self.set_more_shapes()
                if vendor_name == "kunlunxin":
                    if self.op_name in ["cummax"]:
                        additional_shapes = []

                # self.shapes = additional_shapes
                if additional_shapes:
                    self.shapes = list(
                        dict.fromkeys(self.shapes + additional_shapes)
                    )
        except yaml.YAMLError as e:
            raise ValueError(
                f"Shape file '{shape_file_path}' is not"
                f" a valid YAML file. Error: {e}"
            )

    def set_more_shapes(self) -> Optional[List[List[int]]]:
        """Base method (optional to override).

        Returns additional shapes if applicable.
        """
        return None

    def record_shapes(self, *args, **kwargs):
        def deep_parse(item):
            if isinstance(item, torch.Tensor):
                return item.size()
            elif isinstance(item, (int, float, str, torch.dtype)):
                return item
            elif isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return {key: deep_parse(value) for key, value in item.items()}
            return None

        parsed_args = [deep_parse(arg) for arg in args]
        parsed_kwargs = {
            key: deep_parse(value) for key, value in kwargs.items()
        }
        if parsed_args and parsed_kwargs:
            return parsed_args, parsed_kwargs
        return parsed_args if parsed_args else parsed_kwargs

    def init_default_config(self):
        self.set_shapes(self.DEFAULT_SHAPE_FILES)

    def init_user_config(self):
        # TODO: device setting
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        # Prefer a vendor-specific core_shapes.yaml when one is shipped under
        # the backend (e.g. kunlunxin speeds up its run / avoids big-shape
        # timeouts; enflame pins its own shape set). Any backend that drops a
        # core_shapes.yaml into its `_<vendor>` dir is picked up automatically;
        # if the file is absent we fall back to the shared default.
        vendor_shape_file = os.path.join(
            os.path.dirname(__file__),
            "../src/flaggems_sglang/runtime/backend/"
            f"_{vendor_name}/core_shapes.yaml",
        )
        if os.path.isfile(vendor_shape_file):
            Config.shape_file = vendor_shape_file
        self.set_shapes(Config.shape_file)

    def _validate_reference_candidate(self):
        if not reference_mode_enabled():
            return
        if not callable(self.torch_op):
            raise ReferenceContractError(
                f"Operator '{self.op_name}': No predefined reference; "
                "--ref-as-candidate requires an explicitly defined callable "
                "torch_op."
            )
        if self.is_backward:
            raise ReferenceContractError(
                f"Operator '{self.op_name}': --ref-as-candidate does not "
                "support is_backward=True; forward reference invocation "
                "does not prove execution of the backward timer."
            )
        methods = (
            "run",
            "_measure_input",
            "get_latency",
            "_candidate_call",
            "_run_cases",
            "_run_legacy",
            "_benchmark_callable",
            "_time_callable",
        )
        if any(
            getattr(getattr(self, name), "__func__", None)
            is not getattr(Benchmark, name)
            for name in methods
        ):
            raise ReferenceContractError(
                f"Operator '{self.op_name}': custom benchmark execution "
                "does not expose the --ref-as-candidate reference contract."
            )

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def get_latency(self, op, *args, **kwargs):
        fn, xs = self._benchmark_callable(op, *args, **kwargs)
        return self._time_callable(fn, xs)

    def _benchmark_callable(self, op, *args, **kwargs):
        """Build the callable to benchmark (plus the grad inputs for the
        backward case). Split out from timing so the reference-only runner can
        invoke the op exactly once without entering the timing loop."""

        def fn():
            return op(*args, **kwargs)

        xs = None
        if self.is_backward:
            out = fn()
            dout = torch.randn_like(out)
            xs = list(
                filter(
                    lambda x: (torch.is_tensor(x) and x.requires_grad),
                    args,
                )
            )

            def fn():  # noqa: F811
                return torch.autograd.grad(
                    (out,),
                    xs,
                    grad_outputs=(dout,),
                    retain_graph=True,
                )

        return fn, xs

    def _time_callable(self, fn, xs):
        if Config.mode == BenchMode.OPERATOR:
            for i in range(Config.warm_up):
                fn()
            torch_device_fn.synchronize()
            start = time.time()
            for i in range(Config.repetition):
                fn()
            torch_device_fn.synchronize()
            end = time.time()
            latency = (end - start) / Config.repetition * 1000
        elif Config.mode == BenchMode.KERNEL:
            latency = do_bench_latency(
                fn,
                warmup=Config.warm_up,
                rep=Config.repetition,
                grad_to_none=xs if self.is_backward else None,
            )
        elif Config.mode == BenchMode.WRAPPER:
            for i in range(Config.warm_up):
                fn()
            torch_device_fn.synchronize()
            start = time.time()
            for i in range(Config.repetition):
                fn()
            end = time.time()
            latency = (end - start) / Config.repetition * 1000
        else:
            raise ValueError("Undefined Value of Benchmark Mode.")
        # average latency in ms
        return latency

    def get_gbps(self, args, latency=None):
        # """Return the dynamic input iterator for each Operator."""
        raise NotImplementedError(
            "Each Benchmark must implement its own input iterator."
        )

    def get_tflops(self, op, *args, **kwargs):
        """Not really implemented; serves as a
        placeholder for future development."""
        from torch.utils.flop_counter import FlopCounterMode

        def fn():
            return op(*args, **kwargs)

        with FlopCounterMode(display=False) as flop_counter:
            fn()
        return flop_counter.get_total_flops()

    def get_input_iter(self, dtype) -> Generator:
        # """Return the dynamic input iterator for each Operator."""
        raise NotImplementedError(
            "Each Benchmark must implement its own input iterator."
        )

    def get_inputs(self, dtype):
        if self._input_iter is None:
            self._input_iter = self.get_input_iter(dtype)
        try:
            return next(self._input_iter)
        except StopIteration:
            return None

    def unpack_to_args_kwargs(self, input_tuple: Tuple[Any, ...]):
        args = []
        kwargs = {}
        for item in input_tuple:
            if (
                isinstance(item, torch.Tensor)
                or isinstance(item, (int, float))
                or item is None
                or isinstance(item, (list, tuple))
                or isinstance(item, torch.dtype)
            ):
                args.append(item)
            elif isinstance(item, dict):
                kwargs.update(item)
        if self.is_backward:
            args = [
                (
                    a.clone().requires_grad_()  # type: ignore[union-attr]
                    if torch.is_tensor(a) and torch.is_floating_point(a)
                    else a
                )
                for a in args
            ]
        return args, kwargs

    # ---- Case API (tensor-free listing + exact replay) ------------------
    # Operators opt in by overriding get_case_iter / build_inputs. Until an
    # op does, supports_cases() is False and the legacy run loop is used, so
    # behavior is unchanged for every op that has not been migrated.

    def supports_cases(self) -> bool:
        return (
            type(self).get_case_iter is not Benchmark.get_case_iter
            and type(self).build_inputs is not Benchmark.build_inputs
        )

    def _case_id(self, dtype, ordinal: int) -> str:
        nodeid = getattr(Config, "current_nodeid", None)
        if not nodeid:
            raise RuntimeError(
                "Benchmark case IDs require an active pytest nodeid."
            )
        dtype_name = str(dtype)
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name[len("torch.") :]
        local_id = f"{Config.bench_level.value}::{dtype_name}::{ordinal}"
        return f"{nodeid}::{local_id}"

    def _case_from_plan(
        self, dtype, ordinal: int, plan: BenchmarkCasePlan
    ) -> BenchmarkCaseSpec:
        return BenchmarkCaseSpec(
            case_id=self._case_id(dtype, ordinal),
            ordinal=ordinal,
            dtype=dtype,
            shape=plan.shape,
            params=plan.params,
            builder_args=(plan,),
        )

    def get_case_iter(self, dtype) -> Generator:
        """Return lightweight cases without allocating tensor inputs."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide benchmark cases."
        )

    def build_inputs(self, case: BenchmarkCaseSpec):
        """Build the exact legacy input tuple for one selected case."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot materialize benchmark case specs."
        )

    def _collect_cases(self) -> Tuple[BenchmarkCaseSpec, ...]:
        if not self.supports_cases():
            raise ValueError(
                f"Operator '{self.op_name}' does not support case listing "
                "yet."
            )
        cases = tuple(
            case
            for dtype in self.to_bench_dtypes
            for case in self.get_case_iter(dtype)
        )
        if not cases:
            raise ValueError(
                f"Operator '{self.op_name}' generated no benchmark cases."
            )
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Operator '{self.op_name}' generated duplicate case IDs."
            )
        return cases

    def list_cases(self, initialize: bool = True) -> BenchmarkCaseList:
        if initialize:
            self.init_user_config()
        return BenchmarkCaseList(
            op_name=self.op_name,
            level=Config.bench_level.value,
            cases=self._collect_cases(),
        )

    def _emit_result(self, dtype, metrics):
        result = BenchmarkResult(
            level=Config.bench_level.value,
            op_name=self.op_name,
            dtype=str(dtype),
            mode=Config.mode.value,
            result=metrics,
        )
        print(result)
        result_json = result.to_json()
        emit_record_logger(result_json)
        update_result(self.op_name, json.loads(result_json))
        return result

    def _measure_input(self, input, case_id=None, dtype=None):
        metric = BenchmarkMetrics(case_id=case_id)
        try:
            args, kwargs = self.unpack_to_args_kwargs(input)
            metric.shape_detail = self.record_shapes(*args, **kwargs)
            if "latency_base" in self.to_bench_metrics:
                metric.latency_base = self.get_latency(
                    self.torch_op, *args, **kwargs
                )
            if "latency" in self.to_bench_metrics:
                candidate, dispatch, is_override = self._candidate_call()
                registry = Config.override_registry if is_override else None
                override_key = f"flaggems_sglang.{self.op_name}"
                before_override = (
                    registry.call_counts().get(override_key, 0)
                    if registry
                    else 0
                )
                execution = reference_execution()
                before = execution.candidate_calls if execution else 0
                with dispatch:
                    metric.latency = self.get_latency(
                        candidate, *args, **kwargs
                    )
                if registry is not None:
                    after_override = registry.call_counts().get(
                        override_key, 0
                    )
                    if after_override <= before_override:
                        raise RuntimeError(
                            "benchmark timer did not invoke the injected candidate"
                        )
                    metric.candidate_source = "override"
                if reference_mode_enabled():
                    if execution.candidate_calls <= before:
                        raise ReferenceContractError(
                            f"{case_id or self.op_name}: reference candidate "
                            "timer did not invoke the declared reference."
                        )
                    execution.completed_timings += 1
                    metric.candidate_source = "reference"
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
            if (
                getattr(Config, "base_data", None) is not None
                and metric.latency
            ):
                self._compare_against_base(metric, dtype)
            if "gbps" in self.to_bench_metrics:
                metric.gbps_base = self.get_gbps(
                    args, latency=metric.latency_base
                )
                metric.gbps = self.get_gbps(args, latency=metric.latency)
            if "tflops" in self.to_bench_metrics:
                metric.tflops = (
                    self.get_tflops(self.torch_op, *args, **kwargs)
                    / metric.latency
                    / 1e12
                    * 1e3
                )
                # utilization = metric.tflops / metric.latency / 1e12 * 1e3
        except Exception as e:
            metric.error_msg = str(e)
            pytest.fail(str(e))  # raise exception again
        finally:
            gc.collect()
        return metric

    def _compare_against_base(self, metric, dtype):
        """Fill in the optional 'vs Base' speedup from the --base-data
        baseline. On a miss, record the precise reason. No-op unless
        --base-data was supplied (see conftest._load_base_data)."""
        base_rec, miss_reason = lookup_base_record_ex(
            Config.base_data,
            self.op_name,
            dtype,
            metric.shape_detail,
        )
        base_ms = base_rec.get("latency_ms") if base_rec else None
        if base_ms is not None:
            # Both baseline (latency_ms) and metric.latency are in
            # milliseconds, so the raw ratio is dimensionless. Divide by this
            # chip's scaling factor for the baseline's bottleneck resource to
            # normalize away the hardware peak gap; missing factor -> keep the
            # raw ratio.
            speedup = base_ms / metric.latency
            factor = lookup_scaling_factor(
                Config.base_data,
                vendor_name,
                base_rec.get("bottle_neck_unit"),
                dtype,
            )
            if factor:
                speedup /= factor
            metric.compared_speedup = speedup
            print(
                f"[vs Base] {self.op_name} "
                f"shape={metric.shape_detail} "
                f"base={base_ms:.6f}ms "
                f"gems={metric.latency:.6f}ms "
                f"factor={factor} vs_base={speedup:.3f}"
            )
        else:
            # No baseline match: record and print the precise reason
            # (op / shape / dtype missing) so it can be aggregated later.
            metric.base_miss_reason = miss_reason
            print(f"[vs Base][未匹配] {miss_reason}")

    def _run_legacy(self):
        results = []
        for dtype in self.to_bench_dtypes:
            metrics = []
            for input in self.get_input_iter(dtype):
                metrics.append(self._measure_input(input, dtype=dtype))
            results.append(self._emit_result(dtype, metrics))
        return results

    def _run_cases(self, case_ids: Optional[Collection[str]]):
        cases = self._collect_cases()
        available_ids = [case.case_id for case in cases]
        if reference_mode_enabled():
            reference_execution().benchmark_selection_empty = (
                case_ids is not None
                and not set(case_ids).intersection(available_ids)
            )

        select_all = case_ids is None
        selected = set(case_ids or [])
        Config.available_case_ids.update(available_ids)

        results = []
        executed = []
        for dtype in self.to_bench_dtypes:
            metrics = []
            for case in cases:
                if case.dtype != dtype or (
                    not select_all and case.case_id not in selected
                ):
                    continue
                try:
                    input = self.build_inputs(case)
                except Exception as e:
                    print(
                        f"\033[31mFAILED\033[0m: Operator={self.op_name} "
                        f"case_id={case.case_id} err=<<<{e}>>>"
                    )
                    pytest.fail(str(e))
                metrics.append(
                    self._measure_input(
                        input, case_id=case.case_id, dtype=dtype
                    )
                )
                executed.append(case.case_id)
            if metrics:
                results.append(self._emit_result(dtype, metrics))
        Config.executed_case_ids.update(executed)
        return results

    def _candidate_call(self):
        """Resolve one candidate path for timing, preflight and profiling.

        A live override wins over a previously bound implementation. Without
        an override, preserve the original set_gems-or-reference selection,
        including the historical fallback for falsey callable bindings.
        """
        if reference_mode_enabled():
            return (
                select_candidate(self.gems_op, self.torch_op, self.op_name),
                nullcontext(),
                False,
            )
        registry = getattr(Config, "override_registry", None)
        override = (
            registry.get_override(self.op_name)
            if registry is not None
            else None
        )
        op = (
            override if override is not None else self.gems_op or self.torch_op
        )
        return op, nullcontext(), override is not None

    def _run_preflight_cases(self, case_ids: Optional[Collection[str]]):
        """Check candidate executability, not correctness or performance."""
        if not self.supports_cases():
            raise ValueError(
                f"Operator '{self.op_name}' does not support "
                "--preflight-only yet."
            )
        cases = self._collect_cases()
        if not cases:
            raise ValueError(
                f"Operator '{self.op_name}' has no preflight cases."
            )
        Config.available_case_ids.update(case.case_id for case in cases)
        selected = None if case_ids is None else set(case_ids)
        executed = []
        for case in cases:
            if selected is not None and case.case_id not in selected:
                continue
            record = {
                "operator": self.op_name,
                "nodeid": Config.current_nodeid,
                "case_id": case.case_id,
                "override": False,
                "count": 0,
                "status": "failed",
            }
            Config.preflight_records.append(record)
            try:
                args, kwargs = self.unpack_to_args_kwargs(
                    self.build_inputs(case)
                )
                op, dispatch, record["override"] = self._candidate_call()
                with dispatch:
                    record["count"] = 1
                    op(*args, **kwargs)
                    torch_device_fn.synchronize()
            except Exception as exc:
                record["error"] = str(exc)
                raise
            record["status"] = "passed"
            Config.executed_case_ids.add(case.case_id)
            executed.append(case.case_id)
            del args, kwargs
        return executed

    def _run_profile_cases(self, case_ids: Collection[str]):
        """Run one candidate case with warmup/iterations owned by pytest."""
        if not self.supports_cases():
            raise ValueError(
                f"Operator '{self.op_name}' does not support "
                "--profile-only yet."
            )
        cases = self._collect_cases()
        available = {case.case_id for case in cases}
        Config.available_case_ids.update(available)
        selected = set(case_ids)
        executed = []
        for case in cases:
            if case.case_id not in selected:
                continue
            warmup_input = self.build_inputs(case)
            capture_input = self.build_inputs(case)
            args, kwargs = self.unpack_to_args_kwargs(warmup_input)
            capture_args, capture_kwargs = self.unpack_to_args_kwargs(
                capture_input
            )
            op, dispatch, _ = self._candidate_call()
            with dispatch:
                for _ in range(Config.profile_warmup):
                    op(*args, **kwargs)
                torch_device_fn.synchronize()
                scope = (
                    Config.profile_hook(
                        backend=vendor_name, case_id=case.case_id
                    )
                    if Config.profile_hook is not None
                    else None
                )
                with scope if scope is not None else nullcontext():
                    for _ in range(Config.profile_iterations):
                        op(*capture_args, **capture_kwargs)
                    torch_device_fn.synchronize()
            executed.append(case.case_id)
        Config.executed_case_ids.update(executed)
        return executed

    def _run_reference_cases(self, case_ids: Optional[Collection[str]]):
        """Run the original timing baseline once; never enter Gems
        dispatch."""
        if (
            not self.supports_cases()
            or getattr(self.get_latency, "__func__", None)
            is not Benchmark.get_latency
            or getattr(self._measure_input, "__func__", None)
            is not Benchmark._measure_input
        ):
            Config.reference_records.append(
                {
                    "nodeid": Config.current_nodeid,
                    "operator": self.op_name,
                    "status": "UNSUPPORTED",
                    "reason": (
                        "custom or legacy baseline requires an explicit "
                        "reference runner"
                    ),
                }
            )
            pytest.skip("reference-only is unsupported for this benchmark")
        cases = self._collect_cases()
        if not cases:
            raise ValueError(
                f"Operator '{self.op_name}' has no reference cases."
            )
        Config.available_case_ids.update(case.case_id for case in cases)
        selected = None if case_ids is None else set(case_ids)
        executed = []
        selected_cases = [
            case
            for case in cases
            if selected is None or case.case_id in selected
        ]
        records = [
            {
                **case.to_dict(),
                "nodeid": Config.current_nodeid,
                "operator": self.op_name,
                "count": 0,
                "status": "NOT_RUN",
                "reason": "reference traversal stopped before this case",
            }
            for case in selected_cases
        ]
        Config.reference_records.extend(records)
        for case, record in zip(selected_cases, records):
            record.pop("reason")
            record["status"] = "FAILED"
            if getattr(Config, "skip_native", False):
                record.update(
                    status="SKIP",
                    reason=getattr(
                        Config, "native_baseline_skip_reason", None
                    ),
                )
                continue
            args = kwargs = fn = grad_inputs = None
            try:
                record["stage"] = "build_inputs"
                args, kwargs = self.unpack_to_args_kwargs(
                    self.build_inputs(case)
                )
                record["stage"] = "prepare_reference"
                fn, grad_inputs = self._benchmark_callable(
                    self.torch_op, *args, **kwargs
                )
                record["stage"] = "invoke"
                record["count"] = 1
                started = time.perf_counter()
                fn()
                record["stage"] = "synchronize"
                torch_device_fn.synchronize()
                # Readiness records a real, synchronized single invocation.
                # This is not the warmed benchmark performance denominator.
                record["latency_ms"] = (time.perf_counter() - started) * 1000
            except pytest.skip.Exception as error:
                record.update(status="SKIP", reason=str(error))
                raise
            except BaseException as error:
                record["failure"] = reference_failure(error)
                if (
                    not isinstance(error, Exception)
                    or record["stage"] == "synchronize"
                    or record["failure"]["category"] == "UNKNOWN"
                ):
                    raise
                # Only explicit capability errors can continue, and never if
                # queued accelerator work reports an additional failure.
                try:
                    torch_device_fn.synchronize()
                except BaseException as sync_error:
                    record["recovery_failure"] = reference_failure(sync_error)
                    raise
                continue
            finally:
                del fn, grad_inputs, args, kwargs
            record["status"] = "PASSED"
            Config.executed_case_ids.add(case.case_id)
            executed.append(case.case_id)
        failures = sum(record["status"] == "FAILED" for record in records)
        if failures:
            pytest.fail(
                f"{failures} reference cases failed; see per-case reference "
                "report",
                pytrace=False,
            )
        return executed

    def run(self, case_ids: Optional[Collection[str]] = None):
        if Config.query:
            self.init_default_config()
            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc=self.shape_desc,
            )
            print(attri)
            emit_record_logger(attri.to_dict())
            return

        self.init_user_config()
        self._validate_reference_candidate()
        if reference_mode_enabled() and "latency" not in self.to_bench_metrics:
            raise ReferenceContractError(
                "--ref-as-candidate requires the candidate latency metric."
            )
        configured_case_ids = getattr(Config, "case_ids", None)
        selection_requested = (
            case_ids is not None or configured_case_ids is not None
        )
        selected_case_ids = (
            case_ids if case_ids is not None else configured_case_ids
        )

        probe = getattr(
            getattr(Config, "override_registry", None), "_dummy_probe", None
        )
        if probe is not None:
            return probe.run_benchmark(self, selected_case_ids)

        if getattr(Config, "list_cases", False):
            if selected_case_ids:
                raise ValueError(
                    "--list-cases cannot be combined with --case-id."
                )
            case_list = self.list_cases(initialize=False)
            update_case_list(case_list.to_dict())
            print(
                f"Listed {len(case_list.cases)} benchmark cases for "
                f"{self.op_name}."
            )
            return case_list

        if getattr(Config, "reference_only", False):
            return self._run_reference_cases(selected_case_ids)

        if getattr(Config, "preflight_only", False):
            return self._run_preflight_cases(selected_case_ids)

        if getattr(Config, "profile_only", False):
            return self._run_profile_cases(selected_case_ids or [])

        if selection_requested and not self.supports_cases():
            raise ValueError(
                f"Operator '{self.op_name}' does not support --case-id yet."
            )
        if self.supports_cases():
            return self._run_cases(selected_case_ids)
        return self._run_legacy()


class GenericBenchmark(Benchmark):
    """
    A generic benchmark class for most of the operations.

    This class extends the Benchmark base class.
    It allows users to specify custom input functions
    and shapes, making it suitable for a wide range
    of tensor operations including both unary and
    binary operations.

    Usage example:
        benchmark = GenericBenchmark(
            op_name="add",
            torch_op=torch.add,
            input_fn=binary_input_fn,
        )
        benchmark.run()
    """

    def __init__(
        self,
        *args,
        input_fn=None,
        case_fn=None,
        build_inputs_fn=None,
        **kwargs,
    ):
        if input_fn is not None and (
            case_fn is not None or build_inputs_fn is not None
        ):
            raise ValueError(
                "GenericBenchmark accepts either legacy input_fn or the "
                "case_fn/build_inputs_fn pair, not both."
            )
        if (case_fn is None) != (build_inputs_fn is None):
            raise ValueError(
                "GenericBenchmark requires case_fn and build_inputs_fn "
                "together."
            )
        if input_fn is None and case_fn is None:
            raise ValueError(
                "GenericBenchmark requires input_fn or "
                "case_fn/build_inputs_fn."
            )
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.case_fn = case_fn
        self.build_inputs_fn = build_inputs_fn

    def set_more_shapes(self):
        more_shapes_1d = [
            (2**28,),
        ]
        more_shapes_2d = [(10000, 2**i) for i in (0, 8, 16)]
        more_shapes_3d = [(100, 2**i, 100) for i in (0, 8, 16)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        if self.case_fn is not None:
            for case in self.get_case_iter(cur_dtype):
                yield self.build_inputs(case)
            return
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def supports_cases(self) -> bool:
        if (
            type(self).get_case_iter is not GenericBenchmark.get_case_iter
            and type(self).build_inputs is not GenericBenchmark.build_inputs
        ):
            return True
        return (
            self.case_fn is not None
            and type(self).get_input_iter is GenericBenchmark.get_input_iter
        )

    def get_case_iter(self, dtype) -> Generator:
        if self.case_fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide benchmark cases."
            )
        ordinal = 0
        for shape in self.shapes:
            for plan in self.case_fn(shape, dtype):
                if not isinstance(plan, BenchmarkCasePlan):
                    raise TypeError(
                        "case_fn must yield BenchmarkCasePlan instances."
                    )
                yield self._case_from_plan(dtype, ordinal, plan)
                ordinal += 1

    def build_inputs(self, case: BenchmarkCaseSpec):
        if self.build_inputs_fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} cannot materialize benchmark case "
                "specs."
            )
        plan = case.builder_args[0]
        return self.build_inputs_fn(plan, case.dtype, self.device)


class GenericBenchmarkFilterShapes(GenericBenchmark):
    def __init__(self, exclude_dims: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude_dims = exclude_dims

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        if self.exclude_dims is not None:
            return [
                shape for shape in shapes if len(shape) != self.exclude_dims
            ]
        return shapes


class GenericBenchmarkExcluse1D(GenericBenchmarkFilterShapes):
    """
    exclude 1d shapes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=1, *args, **kwargs)


class GenericBenchmarkExcluse3D(GenericBenchmarkFilterShapes):
    """
    exclude 3d shapes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=3, *args, **kwargs)


class GenericBenchmark4DOnly(GenericBenchmarkFilterShapes):
    """
    4d shapes only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=None, *args, **kwargs)

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [shape for shape in shapes if len(shape) == 4]


class GenericBenchmark2DOnly(GenericBenchmarkFilterShapes):
    """
    2d shapes only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=None, *args, **kwargs)

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [shape for shape in shapes if len(shape) == 2]


def generate_tensor_input(shape, dtype, device):
    if dtype in FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    elif dtype in INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    elif dtype in BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            device
        )
    elif dtype in COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)


def binary_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2


def unary_input_fn(shape, cur_dtype, device):
    yield generate_tensor_input(shape, cur_dtype, device),
