# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Source mismatch and output mismatch must leave the entire offline tree intact."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = "src/flaggems_sglang_plugin/apply_operator_patches.py"


class PatchTransactionTests(unittest.TestCase):
    def test_precondition_failure_does_not_change_any_source(self):
        self.run_case("input")

    def test_unexpected_output_does_not_change_any_source(self):
        self.run_case("output")

    def test_qualified_patch_applies_both_files(self):
        self.run_case(None)

    def run_case(self, failure):
        repository = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "isolated_patcher", repository / MODULE_PATH
        )
        patcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patcher)
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            assets = temp / "tool/patches"
            assets.mkdir(parents=True)
            tree = temp / "tree"
            tree.mkdir()
            records = []
            patch = ""
            for name in ("a.py", "b.py"):
                old, new = b"value = 1\n", b"value = 2\n"
                (tree / name).write_bytes(old)
                records.append(
                    dict(
                        path=name,
                        before=hashlib.sha256(old).hexdigest(),
                        after=hashlib.sha256(new).hexdigest(),
                    )
                )
                patch += (
                    "--- a/"
                    + name
                    + "\n+++ b/"
                    + name
                    + "\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
                )
            if failure == "input":
                records[-1]["before"] = "unreviewed"
            if failure == "output":
                records[-1]["after"] = "unreviewed"
            (assets / "manifest.json").write_text(
                json.dumps(dict(test=dict(patch="test.patch", files=records)))
            )
            (assets / "test.patch").write_text(patch)
            patcher.__file__ = str(temp / "tool/apply.py")
            if failure:
                with self.assertRaises(ValueError):
                    patcher.apply(tree, "test")
                self.assertEqual((tree / "a.py").read_bytes(), old)
                self.assertEqual((tree / "b.py").read_bytes(), old)
            else:
                self.assertEqual(patcher.apply(tree, "test"), ["a.py", "b.py"])
                self.assertEqual((tree / "a.py").read_bytes(), new)
                self.assertEqual((tree / "b.py").read_bytes(), new)


if __name__ == "__main__":
    unittest.main()
