#!/usr/bin/env bash

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

# Prepare a GitHub CI environment for FlagGems-sglang, using uv.
#
# Assumes a Python interpreter is already on PATH — in CI that comes from the
# actions/setup-python step, or from a preconfigured venv on a self-hosted
# accelerator runner. This script does not install an interpreter itself.
# Everything else is layered on top, in order:
#
#   1. uv
#   2. build tools
#   3. FlagGems-sglang itself, plus its test dependencies
#   4. sglang
#   5. triton_kernels
#   6. verification
#
# torch and the Triton/FlagTree compiler are NOT installed here. They are
# vendor-specific and come from the runner's base image or an earlier FlagGems
# setup step. Every step above only ever layers on top of them, and verifies
# it left them untouched: a silently swapped torch wheel turns into a confusing
# kernel failure much later in the run.
#
# Usage:
#   .github/scripts/prepare_ci_env.sh
#
# Environment:
#   PYTHON_BIN             interpreter to install into (default: python)
#   UV_VERSION             uv version to install when uv is absent
#   INSTALL_PROJECT        1 (default) / 0 to skip installing this repo
#   EDITABLE_INSTALL       1 (default) editable install of this repo, 0 regular
#   INSTALL_SGLANG         1 (default) / 0 to skip sglang
#   SGLANG_VERSION         sglang version to install
#   SGLANG_INSTALL_DEPS    1 (default) install filtered deps, 0 skip them
#   SGLANG_EXTRA_EXCLUDES  extra space-separated package names to exclude
#   INSTALL_TRITON_KERNELS 1 (default) / 0 to skip triton_kernels
#   TRITON_KERNELS_REF     Triton git ref for triton_kernels
#   TRITON_REPO            Triton git repository URL
#   TRITON_KERNELS_DEPS    0 (default) install with --no-deps, 1 resolve deps
#   REQUIRE_PROJECT        1 (default) require `import flaggems_sglang`
#   REQUIRE_TORCH          1 to require torch, 0 (default) to only report it
#   COLLECT_TESTS          1 (default) run `pytest --collect-only` smoke check
#   UV_INDEX / PIP_INDEX_URL  honoured by uv as usual

set -euo pipefail

# ── Output helpers ───────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { printf ' %b[OK]%b\n' "${GREEN}" "${NC}"; }
fail() { printf ' %b[FAILED]%b\n' "${RED}" "${NC}"; exit 1; }
warn() { printf '%b[WARN]%b %s\n' "${YELLOW}" "${NC}" "$1"; }
step() { printf "\n%s\n" "== $1 =="; }
done_banner() { printf '\n%b%s%b\n' "${GREEN}" "$1" "${NC}"; }

# ── uv behaviour ─────────────────────────────────────────────
# Force uv to copy files into the environment instead of hardlinking from the
# cache. In CI the uv cache and the environment often live on different
# filesystems, where hardlinking silently falls back and can leave a
# partially-populated package (dist-info written, files missing).
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# uv's default HTTP timeout (30s) is too tight on flaky networks and causes
# spurious download failures.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

UV_VERSION="${UV_VERSION:-0.11.22}"

# ── Settings ─────────────────────────────────────────────────
INSTALL_PROJECT="${INSTALL_PROJECT:-1}"
EDITABLE_INSTALL="${EDITABLE_INSTALL:-1}"

INSTALL_SGLANG="${INSTALL_SGLANG:-1}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.19}"
SGLANG_INSTALL_DEPS="${SGLANG_INSTALL_DEPS:-1}"

INSTALL_TRITON_KERNELS="${INSTALL_TRITON_KERNELS:-1}"
TRITON_KERNELS_REF="${TRITON_KERNELS_REF:-v3.6.0}"
TRITON_REPO="${TRITON_REPO:-https://github.com/triton-lang/triton.git}"
TRITON_KERNELS_DEPS="${TRITON_KERNELS_DEPS:-0}"
TRITON_KERNELS_URL="${TRITON_KERNELS_URL:-triton_kernels @ git+${TRITON_REPO}@${TRITON_KERNELS_REF}#subdirectory=python/triton_kernels}"

REQUIRE_PROJECT="${REQUIRE_PROJECT:-1}"
REQUIRE_TORCH="${REQUIRE_TORCH:-0}"
COLLECT_TESTS="${COLLECT_TESTS:-1}"

# Excluded on top of the built-in blacklist in tools/sglang_safe_deps.py.
#
# NVIDIA-only binary wheels: they fail to resolve on non-NVIDIA runners and
# the reference code path never calls into them.
#
# torch-coupled or media binaries: torchcodec is built against one exact torch
# build, and av/decord2 are only reachable through sglang's multimodal input
# path. sglang declares the latter two behind aarch64 environment markers,
# which sglang_safe_deps.py strips — without excluding them here they would be
# installed unconditionally on x86 too.
SGLANG_DEFAULT_EXCLUDES=(
  # NVIDIA-only binary wheels
  sglang-kernel
  sgl-deep-ep
  sgl-deep-gemm
  flash-attn-4
  cuda-python
  cuda-tile
  nvshmem4py-cu13
  nvidia-mathdx
  # torch-coupled / media binaries
  torchcodec
  av
  decord2
)

# ── Repository root ──────────────────────────────────────────
# Derived from this file's location so the script works no matter which
# directory CI invokes it from. ${BASH_SOURCE[0]} is reliable under bash, which
# is what the shebang and the composite action's `shell: bash` give us; it is
# empty when sourced from another shell, where dirname "" would silently
# resolve the root to "/" instead of failing. Validate the result either way.
_self="${BASH_SOURCE[0]:-$0}"
case "${_self}" in
  ""|-*|bash|sh|zsh) _self="" ;;
esac

SCRIPT_DIR=""
if [ -n "${_self}" ] && [ -f "${_self}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${_self}")" && pwd)"
fi

REPO_ROOT=""
if [ -n "${SCRIPT_DIR}" ]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

is_repo() {
  [ -n "$1" ] && [ -f "$1/pyproject.toml" ] && [ -d "$1/src/flaggems_sglang" ]
}

# Reject a root that lacks this repo's markers, then try the obvious
# alternatives rather than letting later steps install into the wrong tree.
if ! is_repo "${REPO_ROOT}"; then
  for _candidate in \
    "${GITHUB_WORKSPACE:-}" \
    "$(git rev-parse --show-toplevel 2>/dev/null || true)" \
    "$(pwd)"; do
    if is_repo "${_candidate}"; then
      REPO_ROOT="${_candidate}"
      break
    fi
  done
fi
unset _self _candidate

if ! is_repo "${REPO_ROOT}"; then
  echo "::error::'${REPO_ROOT}' does not look like a FlagGems-sglang checkout."
  echo "Expected pyproject.toml and src/flaggems_sglang/ there."
  exit 1
fi

# ── Python interpreter ───────────────────────────────────────
# Inside an activated venv (or after actions/setup-python) `python` is the
# right interpreter; fall back to python3 on bare systems.
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python &>/dev/null; then
    PYTHON_BIN="python"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
  else
    echo "::error::no python interpreter on PATH."
    exit 1
  fi
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}" || echo "${PYTHON_BIN}")"
export PYTHON_BIN

# ── uv ───────────────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"

step "uv"
printf "Checking uv ..."
if command -v uv &>/dev/null; then
  printf " %s" "$(uv --version)"
  ok
else
  printf " not found, installing %s ...\n" "${UV_VERSION}"
  # The standalone installer is preferred; fall back to installing uv as a
  # package, which needs no network access beyond the configured index. pip
  # puts that uv next to the interpreter, which is not necessarily on PATH.
  if ! curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh; then
    warn "standalone uv installer failed; installing uv from the package index"
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check -q \
      "uv==${UV_VERSION}" || fail
    _python_dir="$(dirname "${PYTHON_BIN}")"
    export PATH="${_python_dir}:${PATH}"
  fi
  command -v uv &>/dev/null || { printf "uv installation"; fail; }
  printf "Installed %s" "$(uv --version)"
  ok
fi
# Persist PATH for subsequent GitHub Actions steps.
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "${HOME}/.local/bin" >>"${GITHUB_PATH}"
fi

# Always target the resolved interpreter explicitly: uv refuses to install
# without either an active venv or an explicit target, and CI runs it both
# inside a preconfigured venv and against a bare actions/setup-python install.
uv_pip() {
  uv pip "$1" --python "${PYTHON_BIN}" "${@:2}"
}

# ── Runtime-layer guard ──────────────────────────────────────
# triton_kernels is excluded on purpose — it is a separate distribution that
# happens to share the "triton" prefix.
runtime_snapshot() {
  uv_pip freeze 2>/dev/null |
    grep -iE '^(torch|torchaudio|torchvision|triton|triton-[a-z]+|flagtree)([=@ ]|$)' |
    grep -ivE '^triton[-_]kernels' |
    sort || true
}

# Write a uv constraints file pinning the runtime layer to what is already
# installed, and echo its path.
#
# This is the better tool than --no-deps whenever a dependency is pure Python
# and genuinely needs its own transitive deps resolved: uv resolves normally,
# but cannot move torch or the compiler, because a constraint on an already
# satisfied pin leaves it in place. Entries constraints files cannot express
# (direct URL/@ forms, editable installs) are filtered out; those packages are
# still covered by the snapshot diff.
runtime_constraints() {
  local path
  path="$(mktemp)"
  runtime_snapshot | grep -E '^[A-Za-z0-9._-]+==' >"${path}" || true
  echo "${path}"
}

assert_runtime_unchanged() {
  local before="$1" after="$2" what="$3"
  if ! diff -q "${before}" "${after}" >/dev/null 2>&1; then
    echo "::error::${what} modified the torch/triton runtime layer"
    diff -u "${before}" "${after}" || true
    exit 1
  fi
  printf "Runtime layer unchanged by %s" "${what}"
  ok
}

BEFORE="$(mktemp)"
AFTER="$(mktemp)"
CONSTRAINTS="$(mktemp)"
DEPS_FILE="$(mktemp)"
trap 'rm -f "${BEFORE}" "${AFTER}" "${CONSTRAINTS}" "${DEPS_FILE}"' EXIT

# ── Environment report ───────────────────────────────────────
step "Environment"
printf "Repo root: %s\n" "${REPO_ROOT}"
printf "Python:    %s (%s)\n" "$("${PYTHON_BIN}" --version)" "${PYTHON_BIN}"
printf "uv:        %s (%s)\n" "$(uv --version)" "$(command -v uv)"
printf "Index:     %s\n" "${UV_INDEX:-${PIP_INDEX_URL:-<uv default>}}"

runtime_snapshot >"${BEFORE}"
printf "Runtime layer before install:\n"
if [ -s "${BEFORE}" ]; then
  sed 's/^/  /' "${BEFORE}"
else
  printf "  (no torch/triton packages installed)\n"
fi

# ── Build tools ──────────────────────────────────────────────
step "Build tools"
printf "Installing build tools ..."
uv_pip install -q \
  "setuptools>=64.0,<77" \
  "wheel==0.45.0" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0" \
  "PyYAML==6.0.3" \
  || fail
ok

# ── FlagGems-sglang and its test dependencies ────────────────
if [ "${INSTALL_PROJECT}" = "1" ]; then
  step "FlagGems-sglang"

  # --no-deps keeps this project's generic "torch>=2.6.0" and CUDA-only
  # cupy-cuda12x requirements from replacing the vendor runtime. The test
  # dependencies this repo actually needs are installed explicitly below.
  install_target=("${REPO_ROOT}")
  if [ "${EDITABLE_INSTALL}" = "1" ]; then
    install_target=(-e "${REPO_ROOT}")
  fi

  printf "Installing flaggems_sglang (--no-deps) ..."
  uv_pip install -q --no-build-isolation --no-deps "${install_target[@]}" || fail
  ok

  # Runtime and test dependencies. These are pure Python or have a numpy-only
  # ABI requirement, and their own transitive deps must be resolved: installing
  # pytest-md-report with --no-deps, for instance, leaves pytablewriter missing
  # and its pytest11 entry point then breaks every pytest invocation. So let uv
  # resolve, and hold the runtime layer still with a constraints file instead.
  #
  # scipy is capped below 1.18 because 1.18 requires numpy>=2.0, which breaks
  # the torch/numpy ABI on backends whose torch was built against numpy 1.x.
  # cupy-cuda12x from the project's [test] extra is intentionally left out: it
  # is CUDA-12-only and unused by the test suite.
  CONSTRAINTS="$(runtime_constraints)"
  if [ -s "${CONSTRAINTS}" ]; then
    printf "Pinning the runtime layer via constraints:\n"
    sed 's/^/  /' "${CONSTRAINTS}"
  fi

  printf "Installing runtime and test dependencies ..."
  uv_pip install -q --constraints "${CONSTRAINTS}" \
    "packaging>=24.0" \
    "PyYAML==6.0.3" \
    "sqlalchemy>=1.4.31,<2.1" \
    "pytest>=7.1.0" \
    "pytest-md-report==0.8.0" \
    "coverage==7.13.5" \
    "distro==1.9.0" \
    "numpy>=1.26" \
    "scipy>=1.15.3,<1.18.0" \
    || fail
  ok

  runtime_snapshot >"${AFTER}"
  assert_runtime_unchanged "${BEFORE}" "${AFTER}" "the FlagGems-sglang install"
fi

# ── sglang ───────────────────────────────────────────────────
# FlagGems-sglang uses sglang only as a reference implementation source (e.g.
# sglang.srt.lora.utils.LoRABatchInfo); it never needs sglang's serving stack
# or its CUDA kernel wheels. So sglang goes in with --no-deps, and its
# dependencies are added separately from the filtered list produced by
# tools/sglang_safe_deps.py. That tool already drops the packages the FlagOS
# runtime owns (torch, triton, flashinfer, ...).
if [ "${INSTALL_SGLANG}" = "1" ]; then
  step "sglang ${SGLANG_VERSION}"

  SAFE_DEPS_TOOL="${REPO_ROOT}/tools/sglang_safe_deps.py"
  if [ ! -f "${SAFE_DEPS_TOOL}" ]; then
    echo "::error::${SAFE_DEPS_TOOL} not found."
    exit 1
  fi

  runtime_snapshot >"${BEFORE}"

  printf "Installing sglang==%s (--no-deps) ..." "${SGLANG_VERSION}"
  uv_pip install -q --no-deps "sglang==${SGLANG_VERSION}" || fail
  ok

  if [ "${SGLANG_INSTALL_DEPS}" = "1" ]; then
    exclude_args=()
    for pkg in "${SGLANG_DEFAULT_EXCLUDES[@]}" ${SGLANG_EXTRA_EXCLUDES:-}; do
      exclude_args+=(--exclude "${pkg}")
    done

    printf "Resolving filtered sglang dependencies ..."
    "${PYTHON_BIN}" "${SAFE_DEPS_TOOL}" "sglang==${SGLANG_VERSION}" \
      "${exclude_args[@]}" >"${DEPS_FILE}" || fail
    ok
    printf "Filtered dependency set (%s entries):\n" \
      "$(wc -l <"${DEPS_FILE}" | tr -d ' ')"
    sed 's/^/  /' "${DEPS_FILE}"

    if [ -s "${DEPS_FILE}" ]; then
      # --no-deps again: the filtered list is already the full set we want, and
      # resolving transitively would let a dependency-of-a-dependency pull
      # torch or a CUDA runtime wheel back in.
      printf "Installing filtered sglang dependencies ..."
      uv_pip install -q --no-deps -r "${DEPS_FILE}" || fail
      ok
    else
      warn "Filtered dependency set is empty; installing sglang alone"
    fi
  else
    warn "SGLANG_INSTALL_DEPS=0; sglang installed without its dependencies"
  fi

  runtime_snapshot >"${AFTER}"
  assert_runtime_unchanged "${BEFORE}" "${AFTER}" "the sglang install"

  printf "Checking sglang metadata ..."
  "${PYTHON_BIN}" - "${SGLANG_VERSION}" <<'PY' || fail
import importlib.metadata as md
import sys

expected = sys.argv[1]
try:
    found = md.version("sglang")
except md.PackageNotFoundError:
    print("sglang is not installed")
    sys.exit(1)

if found != expected:
    print(f"sglang version mismatch: installed {found}, expected {expected}")
    sys.exit(1)

print(f" sglang {found}")
PY
  ok

  # `import sglang` is deliberately not asserted. The filtered dependency set
  # omits sglang's CUDA kernel wheels, so deep submodules may not import on
  # every backend. FlagGems-sglang treats that as expected: references fall
  # back to their vendored implementations on ImportError (see
  # src/flaggems_sglang/reference/_lora_batch_utils.py). The verification
  # section below reports the import status without failing the job.
else
  warn "INSTALL_SGLANG=0; skipping sglang"
fi

# ── triton_kernels ───────────────────────────────────────────
if [ "${INSTALL_TRITON_KERNELS}" = "1" ]; then
  step "triton_kernels (${TRITON_KERNELS_REF})"

  if ! command -v git &>/dev/null; then
    echo "::error::git is required to install triton_kernels from a git ref."
    exit 1
  fi

  runtime_snapshot >"${BEFORE}"

  tk_args=(-q)
  if [ "${TRITON_KERNELS_DEPS}" != "1" ]; then
    tk_args+=(--no-deps)
  fi

  printf "uv pip install %s ...\n" "\"${TRITON_KERNELS_URL}\""
  uv_pip install "${tk_args[@]}" "${TRITON_KERNELS_URL}" || fail
  printf "Install finished"
  ok

  runtime_snapshot >"${AFTER}"
  assert_runtime_unchanged "${BEFORE}" "${AFTER}" "the triton_kernels install"

  # uv's exit code only tells us it wrote the dist-info; it does not catch a
  # truncated install where files listed in RECORD never landed on disk. That
  # leaves the import degraded to an empty namespace package.
  printf "Checking triton_kernels metadata ..."
  "${PYTHON_BIN}" - <<'PY' || fail
import csv
import importlib.metadata as md
import pathlib
import sys

try:
    dist = md.distribution("triton_kernels")
except md.PackageNotFoundError:
    print("triton_kernels is not installed")
    sys.exit(1)

# Parse RECORD directly rather than using dist.files: that property runs the
# entries through skip_missing_files(), so it silently drops exactly the
# missing files this check exists to find.
record = dist.read_text("RECORD")
if record is None:
    print("no RECORD in the triton_kernels dist-info")
    sys.exit(1)

base = pathlib.Path(str(dist.locate_file("")))
recorded = [row[0] for row in csv.reader(record.splitlines()) if row]
missing = [name for name in recorded if not (base / name).exists()]
if missing:
    print(f"{len(missing)} recorded file(s) missing, e.g. {missing[:5]}")
    sys.exit(1)

print(f" triton_kernels {dist.version} ({len(recorded)} files)")
PY
  ok

  # triton_kernels imports triton at module import time, which pulls in the
  # compiler and (on some vendor builds) a backend plugin. Report the import
  # result but do not fail on it: on a CPU-only style runner the compiler may
  # refuse to initialise while the package itself is installed correctly.
  printf "Importing triton_kernels ..."
  if "${PYTHON_BIN}" - <<'PY'
import triton_kernels

print(" triton_kernels module:", triton_kernels.__file__)
PY
  then
    ok
  else
    printf "\n"
    warn "import triton_kernels failed; the package is installed but its \
compiler import did not initialise on this runner"
  fi
else
  warn "INSTALL_TRITON_KERNELS=0; skipping triton_kernels"
fi

# ── Verification ─────────────────────────────────────────────
# Report what the prepared environment actually contains, and fail early on
# the things that make a whole test job pointless.
#
# Reported but not required: torch, triton, sglang, triton_kernels, GPU
# availability, pytest collection. These legitimately vary by runner; a missing
# sglang only means some references fall back to their vendored versions.
step "Environment report"

"${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata as md
import platform
import sys

print(f"python           {platform.python_version()} ({sys.executable})")
print(f"platform         {platform.platform()}")
print(f"machine          {platform.machine()}")
print()


def report(name, attr="__version__"):
    """Print one package's status without letting a failure propagate."""
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            print(f"{name:<16} not installed")
        else:
            # Installed but not importable: normal for packages whose deeper
            # submodules need vendor kernels absent on this runner.
            print(f"{name:<16} {version} (installed, import failed: {exc})")
        return
    except Exception as exc:
        # flaggems_sglang runs device detection at import time and raises when
        # no supported accelerator is present.
        print(f"{name:<16} import raised {type(exc).__name__}: {exc}")
        return

    print(f"{name:<16} {getattr(mod, attr, '(no version attribute)')}")


for pkg in ("torch", "triton", "triton_kernels", "sglang", "flaggems_sglang",
            "pytest", "numpy", "scipy"):
    report(pkg)

print()
try:
    import torch
except Exception as exc:
    print(f"torch unavailable: {type(exc).__name__}: {exc}")
else:
    print(f"torch cuda build  {torch.version.cuda}")
    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        print(f"accelerator check raised {type(exc).__name__}: {exc}")
    else:
        print(f"cuda available    {available}")
        if available:
            print(f"device count      {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"  [{index}] {torch.cuda.get_device_name(index)}")
PY

step "Hard requirements"

# Run pytest rather than importing it: startup loads the pytest11 entry points,
# so this also catches a plugin whose own dependencies are missing — which
# `import pytest` happily ignores while every real invocation fails.
printf "pytest runnable ..."
if ! PYTEST_OUT="$("${PYTHON_BIN}" -m pytest --version 2>&1)"; then
  printf "\n%s\n" "${PYTEST_OUT}"
  if "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
    echo "::error::pytest is installed but cannot start. A plugin registered \
in the pytest11 entry points most likely has missing dependencies."
  else
    echo "::error::pytest is not installed. Run this script with \
INSTALL_PROJECT=1, which installs the test dependencies."
  fi
  fail
fi
ok

if [ "${REQUIRE_TORCH}" = "1" ]; then
  printf "torch importable ..."
  "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1 || fail
  ok
fi

if [ "${REQUIRE_PROJECT}" = "1" ]; then
  printf "flaggems_sglang importable ..."
  if ! "${PYTHON_BIN}" - <<'PY'
import flaggems_sglang

print(
    " vendor:", flaggems_sglang.vendor_name,
    "device:", flaggems_sglang.device,
)
PY
  then
    printf "\n"
    echo "::error::flaggems_sglang failed to import. It detects the vendor \
device at import time, so this usually means no supported accelerator is \
visible on this runner."
    fail
  fi
  ok
fi

if [ "${COLLECT_TESTS}" = "1" ]; then
  step "Test collection smoke check"
  COLLECT_LOG="$(mktemp)"
  trap 'rm -f "${BEFORE}" "${AFTER}" "${CONSTRAINTS}" "${DEPS_FILE}" "${COLLECT_LOG}"' EXIT

  # Reported, not required: collection imports every test module, and a module
  # needing an absent optional reference should not fail environment setup.
  if (cd "${REPO_ROOT}" && "${PYTHON_BIN}" -m pytest tests --collect-only -q \
        >"${COLLECT_LOG}" 2>&1); then
    tail -3 "${COLLECT_LOG}"
    printf "Collected tests"
    ok
  else
    printf "\n"
    tail -30 "${COLLECT_LOG}"
    warn "pytest collection reported problems; see the log above"
  fi
fi

done_banner "CI environment ready"
