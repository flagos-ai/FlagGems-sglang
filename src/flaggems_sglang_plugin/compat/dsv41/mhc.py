# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from flag_gems.fused.dsv41.mhc import (
    hc_combine,
    hc_mix_stats,
    hc_mix_stats_sinkhorn_exact,
    hc_split_sinkhorn,
)


def hc_mix_stats_sinkhorn(x, fn, scale, base, hc, iters, rms_eps, hc_eps):
    """Common two-stage mHC projection and Sinkhorn with FP32 intermediates."""
    mixes = hc_mix_stats(x, fn, rms_eps)
    return hc_split_sinkhorn(mixes, scale, base, hc, iters, hc_eps)

__all__ = ["hc_combine", "hc_mix_stats", "hc_mix_stats_sinkhorn", "hc_mix_stats_sinkhorn_exact"]
