# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Apply a source-qualified patch set to an offline SGLang package tree.

All preconditions and patch applicability are checked before modifying files.
Run before starting workers; this utility does not patch live processes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def apply(package_root, profile):
    root = Path(package_root).resolve()
    assets = Path(__file__).parent / "patches"
    manifest = json.loads((assets / "manifest.json").read_text())[profile]
    staged = {}
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid package-relative path")
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["before"]:
            raise ValueError("unreviewed SGLang source: " + str(relative))
        staged[relative] = path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="sglang-source-patch-") as directory:
        temporary = Path(directory)
        for relative, data in staged.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        patch = assets / manifest["patch"]
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(patch.resolve())],
            cwd=temporary,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = {}
        for item in manifest["files"]:
            relative = Path(item["path"])
            outputs[relative] = (temporary / relative).read_bytes()
            if hashlib.sha256(outputs[relative]).hexdigest() != item["after"]:
                raise ValueError("unexpected patch result: " + str(relative))
        # No file is modified until the entire patch has passed validation.
        for relative, data in outputs.items():
            path = root / relative
            if path.read_bytes() != staged[relative]:
                raise RuntimeError("source changed during patch preparation")
        for relative, data in outputs.items():
            (root / relative).write_bytes(data)
    return list(map(str, outputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path, help="directory containing sglang/")
    parser.add_argument("profile")
    args = parser.parse_args()
    print(json.dumps(apply(args.package_root, args.profile)))


if __name__ == "__main__":
    main()
