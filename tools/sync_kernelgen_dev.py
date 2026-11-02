#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Maintain a manual-merge master -> kernelgen-dev pull request."""

import argparse
import json
import os
import re
import subprocess
import sys
from urllib.parse import urlencode

SOURCE = "master"
TARGET = "kernelgen-dev"
TITLE = "Sync master into kernelgen-dev"
MARKER = "<!-- kernelgen-sync:master-to-kernelgen-dev:v1 -->"


class GitHubError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def github_request(method, endpoint, data=None):
    """Use gh's environment/keychain authentication; never pass credentials."""
    command = ["gh", "api", "--method", method, endpoint]
    if data is not None:
        command += ["--input", "-"]
    result = subprocess.run(
        command,
        input=json.dumps(data) if data is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        match = re.search(r"HTTP (\d{3})", result.stderr)
        status = int(match.group(1)) if match else None
        hint = ""
        if status == 403:
            hint = (
                " Check contents: read and pull-requests: write permissions "
                "and Settings > Actions > General > Allow GitHub Actions "
                "to create and approve pull requests."
            )
        raise GitHubError(result.stderr.strip() + hint, status)
    return json.loads(result.stdout)


def pull_request_body(repository, source_sha, target_sha, ahead):
    return f"""{MARKER}
Bring commits from `{SOURCE}` into `{TARGET}`. This PR is maintained by the
daily sync workflow and must be reviewed and merged manually.

Last checked source: `{source_sha}`
Last checked target: `{target_sha}`
Source commits not yet in target: **{ahead}**

[Compare checked commits](https://github.com/{repository}/compare/{target_sha}...{source_sha})

- Use **Create a merge commit** to preserve source ancestry; do not squash.
- Review checks and benchmark logs before merging. The existing benchmark CI
  step records failures in its log without failing the whole job.
- Bot-created PR checks may require **Approve workflows to run** first.
- Do not select **Update branch** or use this PR's web conflict editor: those
  would write development-side changes onto its head branch, `{SOURCE}`.
- If there are conflicts, resolve them on a separate branch based on
  `{TARGET}` and submit a PR to `{TARGET}`.

The workflow never changes either branch, resolves conflicts or merges PRs.
"""


def synchronize(request, repository, *, dry_run=False):
    """Only read branches/compare and create or update one exact-pair PR."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository, flags=re.ASCII):
        raise ValueError("repository must be OWNER/REPOSITORY")
    root = f"repos/{repository}"
    owner = repository.split("/", 1)[0]
    source = request("GET", f"{root}/branches/{SOURCE}")["commit"]["sha"]
    target = request("GET", f"{root}/branches/{TARGET}")["commit"]["sha"]
    comparison = request("GET", f"{root}/compare/{target}...{source}")
    ahead = comparison["ahead_by"]
    if not isinstance(ahead, int) or isinstance(ahead, bool) or ahead < 0:
        raise ValueError("GitHub comparison returned an invalid ahead_by")
    result = {"source": source, "target": target, "ahead": ahead}
    if ahead == 0:
        return dict(result, status="up_to_date")

    def existing_pull_request():
        query = urlencode(
            {
                "state": "open",
                "base": TARGET,
                "head": f"{owner}:{SOURCE}",
                "per_page": 100,
            }
        )
        pulls = request("GET", f"{root}/pulls?{query}")
        matching = [
            pull
            for pull in pulls
            if pull["base"]["ref"] == TARGET
            and pull["head"]["ref"] == SOURCE
            and pull["head"]["repo"]["full_name"].lower() == repository.lower()
        ]
        if len(matching) > 1:
            raise RuntimeError("Multiple open sync PRs; resolve duplicates.")
        return matching[0] if matching else None

    body = pull_request_body(repository, source, target, ahead)
    pull = existing_pull_request()
    if pull is None:
        if dry_run:
            return dict(result, status="would_create")
        try:
            pull = request(
                "POST",
                f"{root}/pulls",
                {"base": TARGET, "head": SOURCE, "title": TITLE, "body": body},
            )
            return dict(result, status="created", url=pull["html_url"])
        except GitHubError as error:
            # A concurrent manual creation may win despite workflow locking.
            # Recover only that race, never turn an API error into success.
            if error.status != 422:
                raise
            pull = existing_pull_request()
            if pull is None:
                raise

    result["url"] = pull["html_url"]
    old_body = pull.get("body") or ""
    if not old_body.startswith(MARKER):
        # An existing human-created exact-pair PR is already live-updated by
        # GitHub. Preserve its description and still avoid a duplicate PR.
        return dict(result, status="existing_manual_pr")
    if old_body == body and pull["title"] == TITLE:
        return dict(result, status="unchanged")
    if dry_run:
        return dict(result, status="would_update")
    request(
        "PATCH",
        f"{root}/pulls/{pull['number']}",
        {"title": TITLE, "body": body},
    )
    return dict(result, status="updated")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("provide --repo OWNER/REPOSITORY or GITHUB_REPOSITORY")
    try:
        result = synchronize(github_request, args.repo, dry_run=args.dry_run)
    except (GitHubError, OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Branch sync failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as output:
            output.write(
                f"### master → kernelgen-dev\n\nStatus: `{result['status']}`. "
                f"Source commits pending: {result['ahead']}.\n\n"
            )
            if "url" in result:
                output.write(f"[Review sync PR]({result['url']})\n\n")
            output.write("Manual merge only; no branch was changed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
