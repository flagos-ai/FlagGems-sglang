"""
flag_dnn - DNN operations implemented with Triton
"""

import torch
from flag_dnn import runtime
from flag_dnn import testing  # noqa: F401
from flag_dnn.ops import (  # noqa: F401
    abs,
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool1d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    add,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    batch_norm_aten,
    clamp,
    cumprod,
    cumsum,
    div,
    eq,
    gelu,
    group_norm,
    layer_norm,
    leaky_relu,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    mean,
    mul,
    ne,
    neg,
    pow,
    prelu,
    prod,
    relu,
    rms_norm,
    silu,
    softmax,
    sqrt,
    sub,
    sum,
    threshold,
    threshold_,
    leaky_relu_,
    hardtanh,
    hardtanh_,
    elu,
    elu_,
    rrelu,
    rrelu_,
    mish,
    softplus,
    softsign,
    softshrink,
    softmin,
    mv,
    mm,
    dot,
    conv1d,
    conv2d,
    hardswish,
    relu6,
    selu,
    selu_,
    glu,
    celu,
    celu_,
    tanh,
    logsigmoid,
    embedding,
    embedding_renorm_,
    cummin,
    cummax,
    lt,
    le,
    gt,
    ge,
)
from flag_dnn.config import aten_patch_list, resolve_user_setting
from flag_dnn.runtime.register import Register

device = runtime.device.name
vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")
registrar = Register
current_work_registrar = None
runtime.replace_customized_ops(globals())

__version__ = "0.1.0"

_FULL_CONFIG = (
    ("abs", abs),
    ("adaptive_avg_pool1d", adaptive_avg_pool1d),
    ("adaptive_avg_pool2d", adaptive_avg_pool2d),
    ("adaptive_avg_pool3d", adaptive_avg_pool3d),
    ("adaptive_max_pool1d", adaptive_max_pool1d),
    ("adaptive_max_pool2d", adaptive_max_pool2d),
    ("adaptive_max_pool3d", adaptive_max_pool3d),
    ("add", add),
    ("avg_pool1d", avg_pool1d),
    ("avg_pool2d", avg_pool2d),
    ("avg_pool3d", avg_pool3d),
    ("batch_norm", batch_norm_aten),
    ("clamp", clamp),
    ("cumprod", cumprod),
    ("cumsum", cumsum),
    ("div", div),
    ("eq", eq),
    ("gelu", gelu),
    ("group_norm", group_norm),
    ("layer_norm", layer_norm),
    ("leaky_relu", leaky_relu),
    ("max_pool1d", max_pool1d),
    ("max_pool2d", max_pool2d),
    ("max_pool3d", max_pool3d),
    ("mean", mean),
    ("mul", mul),
    ("ne", ne),
    ("neg", neg),
    ("pow", pow),
    ("prelu", prelu),
    ("prod", prod),
    ("relu", relu),
    ("rms_norm", rms_norm),
    ("silu", silu),
    ("softmax", softmax),
    ("sqrt", sqrt),
    ("sub", sub),
    ("sum", sum),
    ("threshold", threshold),
    ("threshold_", threshold_),
    ("leaky_relu_", leaky_relu_),
    ("hardtanh", hardtanh),
    ("hardtanh_", hardtanh_),
    ("elu", elu),
    ("elu_", elu_),
    ("rrelu", rrelu),
    ("rrelu_", rrelu_),
    ("mish", mish),
    ("softplus", softplus),
    ("softsign", softsign),
    ("softshrink", softshrink),
    ("softmin", softmin),
    ("mv", mv),
    ("mm", mm),
    ("dot", dot),
    ("conv1d", conv1d),
    ("conv2d", conv2d),
    ("hardswish", hardswish),
    ("relu6", relu6),
    ("selu", selu),
    ("selu_", selu_),
    ("glu", glu),
    ("celu", celu),
    ("celu_", celu_),
    ("tanh", tanh),
    ("log_sigmoid", logsigmoid),
    ("embedding", embedding),
    ("embedding_renorm_", embedding_renorm_),
    ("cummin", cummin),
    ("cummax", cummax),
    ("lt", lt),
    ("le", le),
    ("gt", gt),
    ("ge", ge),
)

FULL_CONFIG_BY_FUNC: dict = {}
for _item in _FULL_CONFIG:
    if not _item or len(_item) < 2:
        continue
    fn = _item[1]
    func_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
    FULL_CONFIG_BY_FUNC.setdefault(func_name, []).append(_item)


def enable(
    lib=aten_lib,
    unused=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register all FlagDNN ops except those explicitly excluded.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        unused: Which ops to skip. Supported forms:
            - list/tuple/set of function names (e.g., ["masked_fill", "mul"]).
            - str path to a YAML file ending with .yml/.yaml containing an
              `exclude:` list.
            - "default" or None: auto-load
              vendor/arch-specific
              runtime/backend/_<vendor>/[<arch>/]
              enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable FlagDNN logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Notes:
        - If the exclude list/YAML resolves to empty, all ops are registered.
    """
    global current_work_registrar
    exclude_ops = resolve_user_setting(unused, "exclude")
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=[],
        user_exclude_ops=exclude_ops,
        cpp_patched_ops=list(set(aten_patch_list)),
        lib=lib,
    )


def only_enable(
    lib=aten_lib,
    include=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register only the specified FlagDNN ops and skip the rest.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        include: Which ops to register. Supported forms:
            - list/tuple/set of function names (e.g., ["rms_norm", "softmax"]).
            - str path to a YAML file ending with .yml/.yaml (expects a list or
              an `include:` key).
            - "default" or None: auto-load
              vendor/arch-specific
              runtime/backend/_<vendor>/[<arch>/]
              only_enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable Flag logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Classic usage:
        - Only register a few ops:
            only_enable(include=["rms_norm", "softmax"])
        - Use vendor default YAML:
            only_enable(include="default")  # or include=None
        - Use a custom YAML:
            only_enable(include="/path/to/only_enable.yaml")

    Notes:
        - If the include list/YAML resolves to empty or none of the names match
          known ops, the function warns and returns without registering.
    """
    import warnings

    include_ops = resolve_user_setting(include, "include")
    if not include_ops:
        warnings.warn(
            "only_enable failed: No include entries"
            " resolved from list or yaml."
        )
        return

    global current_work_registrar
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=include_ops,
        user_exclude_ops=[],
        cpp_patched_ops=list(set(aten_patch_list)),
        full_config_by_func=FULL_CONFIG_BY_FUNC,
        lib=lib,
    )


class use_dnn:
    """
    The 'include' parameter has higher priority than 'exclude'.
    When 'include' is not None, use_dnn will not process 'exclude'.
    """

    def __init__(
        self, exclude=None, include=None, record=False, once=False, path=None
    ):
        self.lib = torch.library.Library("aten", "IMPL")
        self.exclude = (
            exclude if isinstance(exclude, (list, tuple, set, str)) else []
        )
        self.include = (
            include if isinstance(include, (list, tuple, set, str)) else []
        )
        self.registrar = Register
        self.record = record
        self.once = once
        self.path = path

    def __enter__(self):
        if self.include:
            only_enable(
                lib=self.lib,
                include=self.include,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )
        else:
            enable(
                lib=self.lib,
                unused=self.exclude,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        if torch.__version__ >= "2.5":
            self.lib._destroy()
        del self.lib
        del self.exclude
        del self.include
        del self.registrar
        current_work_registrar = None


def all_registered_ops():
    if current_work_registrar is None:
        return []
    return current_work_registrar.get_all_ops()


def all_registered_keys():
    if current_work_registrar is None:
        return []
    return current_work_registrar.get_all_keys()


__all__ = [
    "enable",
    "only_enable",
    "use_dnn",
    "all_registered_ops",
    "all_registered_keys",
]
