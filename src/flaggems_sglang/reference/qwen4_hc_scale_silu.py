# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import torch


def reference(x, hc_count):
    return torch.nn.functional.silu(x / hc_count)
