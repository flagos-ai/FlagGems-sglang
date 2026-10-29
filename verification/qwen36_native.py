# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Version-checked original SGLang entrypoints; no portable fallback."""
import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path

SPEC = json.loads(
    Path(__file__).with_name("qwen36-native-sources.json").read_text()
)
PINNED = {
    "sglang": "0.5.11",
    "sglang-kernel": "0.4.2",
    "flashinfer-python": "0.6.8.post1",
    "flashinfer-cubin": "0.6.8.post1",
    "nvidia-cutlass-dsl": "4.4.2",
    "xgrammar": "0.1.32",
    "flashinfer-jit-cache": "0.6.8.post1+cu130",
    "torch": "2.11.0+cu130",
    "triton": "3.6.0",
}


def versions():
    result = {name: version(name) for name in PINNED}
    for name, expected in PINNED.items():
        if result[name] != expected:
            raise RuntimeError(
                f"Native {name} must be {expected}, found {result[name]}"
            )
    return result


def get_native(operator):
    versions()
    if os.environ.get("SGLANG_PLUGINS") not in (None, "", ","):
        raise RuntimeError(
            "Native collection requires third-party SGLang plugins disabled"
        )
    # A nonempty comma whitelist parses to an empty set; empty string enables all.
    os.environ["SGLANG_PLUGINS"] = ","
    if operator in ("select_experts", "_fused_moe_kernel_sequence"):
        from sglang.srt.server_args import (
            ServerArgs,
            get_global_server_args,
            set_global_server_args_for_scheduler,
        )

        try:
            get_global_server_args()
        except ValueError:
            # Native SGLang configuration API; no model/checkpoint is loaded.
            set_global_server_args_for_scheduler(
                ServerArgs(
                    model_path="dummy",
                    tp_size=4,
                    enable_fused_moe_sum_all_reduce=False,
                )
            )
    entry = SPEC["entries"][operator]
    module = importlib.import_module(entry["module"])
    fn = getattr(module, entry["function"])
    node = next(
        n
        for n in ast.parse(inspect.getsource(fn)).body
        if isinstance(n, ast.FunctionDef)
    )
    digest = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    if digest != entry["ast_sha256"]:
        raise RuntimeError(
            f"{operator}: native source differs from pinned upstream entrypoint"
        )
    if operator.startswith("gemma_") and not module._has_flashinfer:
        raise RuntimeError(
            "Native Gemma FlashInfer dispatch is unavailable; import failed"
        )

    return fn


def source_info(operator):
    entry = SPEC["entries"][operator]
    get_native(operator)
    return dict(
        upstream=SPEC,
        operator=operator,
        versions=versions(),
        installed_module=str(
            Path(importlib.import_module(entry["module"]).__file__).resolve()
        ),
    )


def prepare_inputs(operator, inputs):
    if operator != "select_experts":
        return inputs
    from sglang.srt.layers.moe.topk import TopKConfig

    cfg = inputs["topk_config"]
    if isinstance(cfg, TopKConfig):
        return inputs
    copied = dict(inputs)
    copied["topk_config"] = TopKConfig(
        **{
            field.name: getattr(cfg, field.name)
            for field in dataclasses.fields(TopKConfig)
        }
    )
    return copied
