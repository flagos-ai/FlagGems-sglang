# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Load immutable vendor defaults once, returning independent launch maps."""

from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "tune_configs.yaml"


@lru_cache(maxsize=1)
def _load():
    with _CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("MThreads tuning configuration must be a mapping")
    return config


def tuning_record(op, profile):
    record = _load()[op][profile]
    if not isinstance(record, (dict, list)):
        raise ValueError(f"invalid tuning profile: {op}/{profile}")
    return record


def launch_options(op, profile, overrides=None):
    options = dict(tuning_record(op, profile)["launch"])
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("launch_options must be a mapping")
        options.update(overrides)
    return _validate_options(options)


def _validate_options(options):
    if not isinstance(options, dict):
        raise ValueError("launch options must be a mapping")
    for name, value in options.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"invalid positive integer launch option: {name}")
        if name not in {"num_stages", "GROUP_SIZE_M"} and value & (value - 1):
            raise ValueError(f"launch option must be a power of two: {name}")
    return options


def moe_config(
    config, profile, *, tokens, b_shape, top_k, fuse_swiglu, mul_routed_weight
):
    config = dict(config)
    if profile is None:
        return config
    records = tuning_record("fused_moe_bf16_gemm", profile)
    records = records if isinstance(records, list) else [records]
    actual = dict(
        tokens=tokens,
        b_shape=list(b_shape),
        top_k=top_k,
        fuse_swiglu=fuse_swiglu,
        mul_routed_weight=mul_routed_weight,
    )
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("match", {}), dict
        ):
            raise ValueError("invalid MoE tuning profile")
        matched = True
        for name, value in record.get("match", {}).items():
            if name == "min_tokens":
                matched = matched and tokens >= value
            elif name not in actual:
                raise ValueError(f"unknown MoE tuning condition: {name}")
            else:
                matched = matched and actual[name] == value
        if not matched or any(
            config.get(k) != v for k, v in record.get("baseline", {}).items()
        ):
            continue
        options = _validate_options(dict(record.get("launch", {})))
        if "BLOCK_SIZE_M" in options and options["BLOCK_SIZE_M"] != config.get(
            "BLOCK_SIZE_M"
        ):
            raise ValueError(
                "tuning cannot change routing alignment after metadata preparation"
            )
        config.update(options)
        break
    return config
