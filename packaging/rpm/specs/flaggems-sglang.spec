%global debug_package %{nil}

# Distros that ship pyproject-rpm-macros (Fedora, EL9+) build via the
# %%pyproject_* macro family. Distros without it (openEuler 24.03,
# EL8-family) fall back to a plain pip wheel/install build.
# Capability-detected at parse time, so the build container must have its
# python toolchain installed before rpmbuild runs (Dockerfile.rpm does).
%if %{defined pyproject_wheel}
%global has_pyproject_macros 1
%else
%global has_pyproject_macros 0
%endif

# Filter the auto-generated Requires for torch and triton: the distro
# torch is CPU-only and the distro triton too old; users install both via
# pip (see packaging/INSTALL.md).
# Backslashes are doubled: the macro body is expanded again when rpm reads
# it as a regular expression.
%global __requires_exclude ^python3(\\.[0-9]+)?dist\\((torch|triton)\\)

Name:           python3-flaggems-sglang
# NOTE: version is duplicated across 2 packaging locations - keep in sync:
#   1. this Version: line
#   2. packaging/debian/changelog (latest entry)
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagGems-sglang - Triton operator library for SGLang

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagGems-sglang
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flaggems-sglang-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 64
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
%if %{has_pyproject_macros}
BuildRequires:  pyproject-rpm-macros
%endif
# The pip fallback has no dependency generator; declare the runtime
# modules by hand so both build paths produce the same Requires.
%if !%{has_pyproject_macros}
Requires:       python3-packaging
Requires:       python3-pybind11
Requires:       python3-pyyaml
Requires:       python3-sqlalchemy >= 1.4.31
Requires:       python3-typing-extensions
%endif

%description
FlagGems-sglang is a high-performance operator library written in the
Triton language for the SGLang inference engine. Operators are resolved at
import time through a generic / vendor / architecture registrar, so one
package serves every FlagOS backend. Triton and torch are required at
runtime and are installed separately (see packaging/INSTALL.md).

%prep
%autosetup -n flaggems-sglang-%{version}

%build
%if %{has_pyproject_macros}
%pyproject_wheel
%else
%{__python3} -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
%endif

%install
%if %{has_pyproject_macros}
%pyproject_install
%pyproject_save_files flaggems_sglang
%else
%{__python3} -m pip install --no-deps --no-index --no-warn-script-location \
    --root %{buildroot} --prefix /usr dist/*.whl
%endif

%check
# find_spec smoke test only: importing the module would need torch and
# triton, which are the user's runtime dependencies, not the builder's.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flaggems_sglang'); assert s and s.origin, 'flaggems_sglang not findable'; print('OK: flaggems_sglang at', s.origin)"
# Subpackage completeness: every __init__.py under src/ must be installed
# (same check as packaging/debian/rules).
src_inits="$(cd src && find flaggems_sglang -name __init__.py)"
if [ -z "$src_inits" ]; then
    echo "ERROR: no __init__.py found under src/flaggems_sglang; check would be vacuous"
    exit 1
fi
missing=""
for f in $src_inits; do
    if [ ! -f "%{buildroot}%{python3_sitelib}/$f" ]; then missing="$missing $f"; fi
done
if [ -n "$missing" ]; then
    echo "ERROR: subpackages missing from buildroot:$missing"
    exit 1
fi
echo "OK: all package __init__.py files from src/ present in buildroot"

%if %{has_pyproject_macros}
%files -f %{pyproject_files}
%license LICENSE
%else
%files
%license LICENSE
%{python3_sitelib}/flaggems_sglang/
%{python3_sitelib}/flaggems_sglang-%{version}.dist-info/
%endif

%changelog
* Sat Oct 17 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging.
