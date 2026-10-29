# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline native setup in a new environment; leaves the base untouched."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "verification/qwen36-native-wheels.json").read_text()
    )
    if not args.verify_only:
        if args.venv.exists():
            raise FileExistsError(
                "Choose a new environment; existing ones are never reset"
            )
        if args.wheelhouse is None:
            parser.error("--wheelhouse is required for setup")
        wheels = []
        for item in manifest:
            path = args.wheelhouse / item["filename"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["digests"]["sha256"]:
                raise ValueError(f"Official wheel hash mismatch: {path.name}")
            wheels.append(str(path))
        # The image supplies Torch/CUDA, NVCC, and other v0.5.11 requirements.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                "--system-site-packages",
                str(args.venv),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(args.venv / "bin/python"),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                *wheels,
            ],
            check=True,
        )
    # Fail on a wrong source/version before running operator workloads.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from qwen36_native import get_native, SPEC, versions; "
        "print(versions()); "
        "[print(op, get_native(op).__name__) for op in SPEC['entries']]"
    )
    subprocess.run(
        [
            str(args.venv / "bin/python"),
            "-c",
            script,
            str(ROOT / "verification"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
