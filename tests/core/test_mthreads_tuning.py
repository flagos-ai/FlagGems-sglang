# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the vendor tuning data consumed by public operators."""

import pytest
import yaml

from flaggems_sglang.runtime.backend._mthreads.ops import _tuning


def test_profile_copies_do_not_mutate_defaults():
    options = _tuning.launch_options("deterministic_moe_combine", "m2048")
    options["num_warps"] = 1
    assert (
        _tuning.launch_options("deterministic_moe_combine", "m2048")[
            "num_warps"
        ]
        == 8
    )


@pytest.mark.parametrize("value", [0, -1, True, 3, "8"])
def test_invalid_warp_override_is_rejected(value):
    with pytest.raises(ValueError):
        _tuning.launch_options(
            "deterministic_moe_combine", "m2048", {"num_warps": value}
        )


def test_explicit_launch_override_preserves_other_default_fields():
    options = _tuning.launch_options(
        "deterministic_moe_combine", "m2048", {"num_warps": 4}
    )
    assert options == dict(BLOCK_M=2, BLOCK_DIM=512, num_warps=4, num_stages=1)


def test_moe_profile_matches_shape_and_baseline_without_mutation():
    original = dict(
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1
    )
    args = dict(
        tokens=4,
        b_shape=(256, 512, 2048),
        top_k=8,
        fuse_swiglu=False,
        mul_routed_weight=False,
    )
    selected = _tuning.moe_config(original, "m4_up", **args)
    assert selected == dict(original, BLOCK_SIZE_N=64)
    assert original["BLOCK_SIZE_N"] == 32
    assert (
        _tuning.moe_config(original, "m4_up", **dict(args, tokens=5))
        == original
    )
    changed_baseline = dict(original, BLOCK_SIZE_K=32)
    assert (
        _tuning.moe_config(changed_baseline, "m4_up", **args)
        == changed_baseline
    )


def test_moe_tuning_cannot_change_existing_routing_alignment(
    tmp_path, monkeypatch
):
    path = tmp_path / "tuning.yaml"
    path.write_text(
        yaml.safe_dump(
            {"fused_moe_bf16_gemm": {"bad": {"launch": {"BLOCK_SIZE_M": 32}}}}
        )
    )
    monkeypatch.setattr(_tuning, "_CONFIG_PATH", path)
    _tuning._load.cache_clear()
    try:
        with pytest.raises(ValueError, match="routing alignment"):
            _tuning.moe_config(
                {"BLOCK_SIZE_M": 16},
                "bad",
                tokens=4,
                b_shape=(256, 512, 2048),
                top_k=8,
                fuse_swiglu=False,
                mul_routed_weight=False,
            )
    finally:
        _tuning._load.cache_clear()


def test_group_count_can_be_a_non_power_of_two():
    assert _tuning._validate_options({"GROUP_SIZE_M": 3}) == {
        "GROUP_SIZE_M": 3
    }
