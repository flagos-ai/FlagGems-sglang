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

"""RNG generator factory.

Some vendor backends do not implement device-side RNG for the explicit
``generator=`` dispatch path. The Enflame (GCU) backend, for example, rejects
``torch.Generator(device="gcu")`` with ``RuntimeError: Expected a 'cpu' device
type for generator but found 'gcu'`` even though the device RNG kernels exist
internally. Passing a CPU-seeded generator with ``device="gcu"`` in the factory
call works and yields a sequence identical to the pure CPU MT19937 stream.

Usage::

    from flaggems_sglang.utils import create_generator

    g = create_generator(device)          # instead of torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(m, n, generator=g, device=device)

``create_generator`` always returns a generator object supporting
``manual_seed``; callers never need to special-case the vendor.
"""

from __future__ import annotations

import torch

from flaggems_sglang import vendor_name

# Vendors whose backend rejects explicit device generators. Keep this a
# tuple so future backends can be added without touching the factory.
_CPU_GENERATOR_VENDORS = ("enflame",)


def _needs_cpu_generator(device) -> bool:
    """True when ``device``-side generators must be emulated via CPU.

    On vendors listed in ``_CPU_GENERATOR_VENDORS`` every non-CPU device
    resolves to a CPU generator, because their backend rejects explicit
    device generators. CPU devices always use a CPU generator anyway.
    """
    if vendor_name not in _CPU_GENERATOR_VENDORS:
        return False
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return dev.type != "cpu"


def create_generator(device=None, *, seed=None):
    """Return a generator usable with ``generator=`` on this backend.

    Args:
        device: device the caller intends to place tensors on. May be a
            ``torch.device``, an int index, or a string such as ``"gcu"``.
            Ignored when ``vendor_name`` is not in the CPU-generator list.
        seed: optional seed applied via ``manual_seed`` before returning.

    Returns:
        A ``torch.Generator``. On CUDA-style backends this is the device
        generator (``torch.Generator(device=device)``); on backends listed in
        ``_CPU_GENERATOR_VENDORS`` it is a CPU generator, which those stacks
        accept for ``randn/randint/randperm(..., generator=..., device=...)``.
    """
    if _needs_cpu_generator(device):
        gen = torch.Generator(device="cpu")
    else:
        gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def randn(*shape, generator=None, device=None, **kwargs):
    """Convenience randn that routes through :func:`create_generator`."""
    if generator is None:
        generator = create_generator(device)
    return torch.randn(*shape, generator=generator, device=device, **kwargs)


def randint(low, high, *shape, generator=None, device=None, **kwargs):
    """Convenience randint that routes through :func:`create_generator`."""
    if generator is None:
        generator = create_generator(device)
    return torch.randint(
        low, high, shape, generator=generator, device=device, **kwargs
    )


def randperm(n, generator=None, device=None, **kwargs):
    """Convenience randperm that routes through :func:`create_generator`."""
    if generator is None:
        generator = create_generator(device)
    return torch.randperm(n, generator=generator, device=device, **kwargs)


__all__ = ["create_generator", "randn", "randint", "randperm"]
