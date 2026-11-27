# SPDX-License-Identifier: Apache-2.0
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
# Adapted from the native SGLang MUSA JIT loader; see NOTICE for provenance.
"""Build the vendored communication sources with the active MUSA toolkit."""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Sequence

import tvm_ffi
import tvm_ffi.cpp.extension as tvm_ffi_ext
from tvm_ffi.libinfo import (
    find_dlpack_include_path,
    find_include_path,
    find_libtvm_ffi,
)
from tvm_ffi.utils import FileLock

_CSRC_DIR = Path(__file__).resolve().parent
_CACHE_DIR = Path(
    os.getenv(
        "SGLANG_MUSA_JIT_CACHE_DIR", Path.home() / ".cache" / "sglang_musa_jit"
    )
)


@functools.cache
def _musa_home() -> Path:
    return Path(tvm_ffi_ext._find_musa_home())


def _mcc() -> str:
    return str(_musa_home() / "bin" / "mcc")


@functools.cache
def _musa_arch_names() -> tuple[str, ...]:
    import torch

    props = torch.musa.get_device_properties(torch.musa.current_device())
    return (f"mp_{int(props.major)}{int(props.minor)}",)


def _musa_arch_flags() -> list[str]:
    return [f"--offload-arch={arch}" for arch in _musa_arch_names()]


def _escape(path: Path) -> str:
    return path.resolve().as_posix().replace(":", "$:")


def _write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _object_path(build_dir: Path, source: Path) -> Path:
    suffix = ".musa.o" if source.suffix in {".mu", ".cu"} else ".o"
    return build_dir / f"{source.parent.name}_{source.stem}{suffix}"


def _sources_digest(sources: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    dependencies = set(sources) | set(_CSRC_DIR.rglob("*.h"))
    dependencies.add(_CSRC_DIR / "distributed/custom_all_reduce.mu")
    for source in sorted(dependencies):
        digest.update(source.relative_to(_CSRC_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@functools.cache
def _compiler_version() -> str:
    return subprocess.check_output([_mcc(), "--version"], text=True)


def _tvm_ffi_version() -> str:
    return str(getattr(tvm_ffi, "__version__", "unknown"))


def _module_cache_name(
    name: str,
    sources: Sequence[Path],
    extra_cflags: Sequence[str],
    extra_musa_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(_sources_digest(sources).encode())
    digest.update(_compiler_version().encode())
    for flags in (extra_cflags, extra_musa_cflags, extra_ldflags):
        for flag in flags:
            digest.update(flag.encode())
            digest.update(b"\0")
    suffix = f"__src_{digest.hexdigest()[:16]}__tvmffi_{_tvm_ffi_version()}"
    return f"{name}{suffix}"


def _musa_arch_cache_dir() -> Path:
    return _CACHE_DIR / "_".join(_musa_arch_names())


def musa_jit_build_dir(name: str) -> Path:
    return _musa_arch_cache_dir() / name


def _ninja_content(
    name: str,
    sources: Sequence[Path],
    extra_cflags: Sequence[str],
    extra_musa_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
) -> str:
    source_digest_flag = (
        f"-DSGLANG_MUSA_JIT_SOURCE_HASH=0x{_sources_digest(sources)}"
    )
    include_flags = [
        f"-I{find_include_path()}",
        f"-I{find_dlpack_include_path()}",
        f"-I{_musa_home() / 'include'}",
    ]
    cflags = [
        "-std=c++17",
        "-fPIC",
        "-O3",
        source_digest_flag,
        *extra_cflags,
        *include_flags,
    ]
    musa_cflags = [
        "-std=c++17",
        "-fPIC",
        "-O3",
        "-ffast-math",
        *_musa_arch_flags(),
        source_digest_flag,
        *extra_musa_cflags,
        *include_flags,
    ]
    libtvm = Path(find_libtvm_ffi())
    ldflags = [
        "-shared",
        f"-L{libtvm.parent}",
        "-ltvm_ffi",
        f"-L{_musa_home() / 'lib'}",
        "-lmusa",
        "-lmusart",
        *extra_ldflags,
    ]

    build_dir = musa_jit_build_dir(name)
    lines = [
        "ninja_required_version = 1.3",
        f"cxx = {os.environ.get('CXX', 'c++')}",
        f"mcc = {_mcc()}",
        f"cflags = {' '.join(cflags)}",
        f"musa_cflags = {' '.join(musa_cflags)}",
        f"ldflags = {' '.join(ldflags)}",
        "",
        "rule compile",
        "  depfile = $out.d",
        "  deps = gcc",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out",
        "",
        "rule compile_musa",
        "  depfile = $out.d",
        "  deps = gcc",
        "  command = $mcc -MD -MF $out.d $musa_cflags -c $in -o $out",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]
    objects: list[str] = []
    for source in sources:
        obj = _object_path(build_dir, source)
        objects.append(_escape(obj))
        rule = "compile_musa" if source.suffix in {".mu", ".cu"} else "compile"
        lines.append(f"build {objects[-1]}: {rule} {_escape(source)}")
    output_so = _escape(build_dir / f"{name}.so")
    lines += [
        "",
        f"build {output_so}: link {' '.join(objects)}",
        f"default {output_so}",
        "",
    ]
    return "\n".join(lines)


@functools.lru_cache(maxsize=16)
def load_musa_jit(
    name: str,
    sources: tuple[str, ...],
    extra_cflags: tuple[str, ...] = (),
    extra_musa_cflags: tuple[str, ...] = (),
    extra_ldflags: tuple[str, ...] = (),
):
    resolved_sources = tuple(
        (_CSRC_DIR / source).resolve() for source in sources
    )
    module_name = _module_cache_name(
        name,
        resolved_sources,
        extra_cflags,
        extra_musa_cflags,
        extra_ldflags,
    )
    build_dir = musa_jit_build_dir(module_name)
    ninja_path = build_dir / "build.ninja"
    so_path = build_dir / f"{module_name}.so"
    lock_path = _musa_arch_cache_dir() / "tmp" / f"{module_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        build_dir.mkdir(parents=True, exist_ok=True)
        _write_if_different(
            ninja_path,
            _ninja_content(
                module_name,
                resolved_sources,
                extra_cflags,
                extra_musa_cflags,
                extra_ldflags,
            ),
        )
        subprocess.run(
            ["ninja", "-C", str(build_dir), "-f", str(ninja_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return tvm_ffi.load_module(str(so_path))


@functools.lru_cache(maxsize=16)
def load_musa_pybind(
    name: str,
    sources: tuple[str, ...],
    extra_cflags: tuple[str, ...] = (),
    extra_musa_cflags: tuple[str, ...] = (),
    extra_ldflags: tuple[str, ...] = (),
):
    from torch_musa.utils.musa_extension import load

    resolved_sources = tuple(
        str((_CSRC_DIR / source).resolve()) for source in sources
    )
    build_dir = musa_jit_build_dir(name)
    lock_path = _musa_arch_cache_dir() / "tmp" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    libtvm = Path(find_libtvm_ffi())
    include_paths = [
        str(find_include_path()),
        str(find_dlpack_include_path()),
        str(_musa_home() / "include"),
    ]
    ldflags = [
        f"-L{libtvm.parent}",
        "-ltvm_ffi",
        f"-L{_musa_home() / 'lib'}",
        "-lmusa",
        "-lmusart",
        *extra_ldflags,
    ]
    with FileLock(str(lock_path)):
        return load(
            name=name,
            sources=list(resolved_sources),
            extra_cflags=[
                "-O3",
                "-std=c++17",
                *extra_cflags,
            ],
            extra_musa_cflags=[
                "-O3",
                "-std=c++17",
                "-ffast-math",
                *_musa_arch_flags(),
                *extra_musa_cflags,
            ],
            extra_ldflags=ldflags,
            extra_include_paths=include_paths,
            build_directory=str(build_dir),
            verbose=os.environ.get("SGLANG_MUSA_JIT_VERBOSE") == "1",
            with_musa=True,
            is_python_module=True,
        )
