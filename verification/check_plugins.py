# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay immutable NVIDIA datasets under the documented plugin names."""

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = json.loads((ROOT / "plugin_datasets.json").read_text())


def check_plugins(
    operators, *, device=None, cache_dir=None, config=None, checker=None
):
    """Keep stored IDs/objects intact and bind each result to its plugin name."""
    operators = list(operators)
    if not operators or len(set(operators)) != len(operators):
        raise ValueError("Select distinct plugin operators")
    if set(operators) - DATASETS.keys():
        raise ValueError("Unknown plugin operator")
    if config is None:
        from op_verify.config import load_config

        config = load_config()
    if checker is None:
        from op_verify.check_cmd import run_check

        checker = run_check
    groups = {}
    for operator in operators:
        dataset = DATASETS[operator]
        groups.setdefault(dataset["prefix"], []).append(operator)
    output = dict(
        results={}, summary=dict(total_pass=0, total_fail=0, total_skip=0)
    )
    for prefix, names in groups.items():
        local = copy.copy(config)
        # Explicit assignment avoids Config.__post_init__ reapplying an
        # unrelated global prefix over the immutable per-dataset namespace.
        local.ks3_prefix = prefix
        local.problems_dir = str(ROOT / "legacy")
        group = DATASETS[names[0]]["cache_group"]
        local.cache_dir = str(Path(cache_dir or config.cache_dir) / group)
        problems = [DATASETS[name]["problem"] for name in names]
        report = checker(local, problems=problems, device=device, isolate=True)
        if set(report.get("results", {})) != set(problems):
            raise RuntimeError(
                "Dataset replay did not return every selected problem"
            )
        for name, problem in zip(names, problems):
            result = report["results"][problem]
            if len(result.get("cases", [])) != DATASETS[name]["cases"]:
                raise RuntimeError(
                    f"{name}: replay did not cover the fixed dataset cases"
                )
            output["results"][name] = dict(
                report["results"][problem],
                dataset_problem=problem,
                dataset_prefix=prefix,
            )
        for key in output["summary"]:
            output["summary"][key] += report["summary"][key]
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator", action="append", choices=sorted(DATASETS)
    )
    parser.add_argument("--device")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show mappings without accessing KS3",
    )
    args = parser.parse_args(argv)
    names = args.operator or list(DATASETS)
    if args.list:
        print(json.dumps({name: DATASETS[name] for name in names}, indent=2))
        return 0
    report = check_plugins(names, device=args.device, cache_dir=args.cache_dir)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return int(
        bool(
            report["summary"]["total_fail"] or report["summary"]["total_skip"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
