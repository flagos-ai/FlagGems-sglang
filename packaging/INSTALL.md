# Installing python3-flaggems-sglang

After `apt install python3-flaggems-sglang` (Debian/Ubuntu) or
`dnf install python3-flaggems-sglang` (Fedora/openEuler), install the ML
runtime separately. It is intentionally **not** declared as a hard
Depends/Requires because the distro versions are CPU-only (torch) or too
old (triton) for accelerator workloads.

FlagGems-sglang needs Triton and torch at runtime:

  - `apt install python3-flagtree-nvidia` (FlagOS-provided, satisfies the
    triton namespace), OR `pip install triton` matching your stack;
  - torch via pip matching your accelerator stack, for example

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

A `venv` with `--system-site-packages` keeps the pip-installed runtime next
to the packaged module:

```bash
python3 -m venv --system-site-packages ~/.venv/flagos
source ~/.venv/flagos/bin/activate
# then the pip install lines above
```
