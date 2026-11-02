# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-only behavioral checks for the manual-merge branch sync helper."""

import copy
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

SOURCE_PATH = Path(__file__).resolve().parents[1] / "sync_kernelgen_dev.py"
SPEC = importlib.util.spec_from_file_location(
    "branch_sync_under_test", SOURCE_PATH
)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

REPOSITORY = "flagos-ai/FlagGems-sglang"
API_ROOT = "repos/" + REPOSITORY
SOURCE_SHA = "a" * 40
TARGET_SHA = "b" * 40


def pull(number=12, *, body="Human description", repository=REPOSITORY):
    return {
        "number": number,
        "title": "An existing pull request",
        "body": body,
        "html_url": f"https://github.com/{REPOSITORY}/pull/{number}",
        "head": {"ref": sync.SOURCE, "repo": {"full_name": repository}},
        "base": {"ref": sync.TARGET},
    }


class MemoryGitHub:
    """A restrictive API fake: branch changes and merges are always errors."""

    def __init__(self, ahead=1, source=SOURCE_SHA, target=TARGET_SHA):
        self.ahead = ahead
        self.source = source
        self.target = target
        self.pulls = []
        self.calls = []
        self.creation_error = None
        self.concurrent_pull = None

    @property
    def writes(self):
        return [call for call in self.calls if call[0] != "GET"]

    def __call__(self, method, endpoint, data=None):
        self.calls.append((method, endpoint, copy.deepcopy(data)))
        if method == "GET":
            branches = {
                f"{API_ROOT}/branches/{sync.SOURCE}": self.source,
                f"{API_ROOT}/branches/{sync.TARGET}": self.target,
            }
            if endpoint in branches:
                return {"commit": {"sha": branches[endpoint]}}
            if endpoint == f"{API_ROOT}/compare/{self.target}...{self.source}":
                return {"ahead_by": self.ahead}
            parsed = urlsplit(endpoint)
            if parsed.path == f"{API_ROOT}/pulls":
                query = parse_qs(parsed.query)
                assert query["state"] == ["open"]
                assert query["base"] == [sync.TARGET]
                assert query["head"] == ["flagos-ai:" + sync.SOURCE]
                return copy.deepcopy(self.pulls)
        if method == "POST" and endpoint == f"{API_ROOT}/pulls":
            assert data["head"] == sync.SOURCE
            assert data["base"] == sync.TARGET
            if self.creation_error is not None:
                if self.concurrent_pull is not None:
                    self.pulls.append(self.concurrent_pull)
                raise self.creation_error
            created = pull(body=data["body"])
            created["title"] = data["title"]
            self.pulls.append(created)
            return copy.deepcopy(created)
        for existing in self.pulls:
            if method == "PATCH" and endpoint == (
                f"{API_ROOT}/pulls/{existing['number']}"
            ):
                assert set(data) == {"title", "body"}
                existing.update(data)
                return copy.deepcopy(existing)
        raise AssertionError(f"Unexpected API operation: {method} {endpoint}")


class SynchronizeTests(unittest.TestCase):
    def test_real_commit_graphs_choose_only_pending_master_commits(self):
        with tempfile.TemporaryDirectory() as directory:

            def git(*args):
                return subprocess.check_output(
                    [
                        "git",
                        "-c",
                        "user.name=Sync Test",
                        "-c",
                        "user.email=sync-test@example.invalid",
                        "-c",
                        "commit.gpgsign=false",
                        "-c",
                        "core.hooksPath=/dev/null",
                        *args,
                    ],
                    cwd=directory,
                    text=True,
                    stderr=subprocess.PIPE,
                ).strip()

            git("init", "--quiet")
            git("commit", "--allow-empty", "-m", "common base", "--quiet")
            ancestor = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-m", "master change", "--quiet")
            source = git("rev-parse", "HEAD")
            git("checkout", "--detach", ancestor, "--quiet")
            git(
                "commit",
                "--allow-empty",
                "-m",
                "development change",
                "--quiet",
            )
            target = git("rev-parse", "HEAD")
            cases = [
                ("identical", ancestor, ancestor, 0),
                ("development only", ancestor, target, 0),
                ("master only", source, ancestor, 1),
                ("diverged", source, target, 1),
            ]
            for name, source_sha, target_sha, expected in cases:
                with self.subTest(graph=name):
                    ahead = int(
                        git(
                            "rev-list",
                            "--count",
                            f"{target_sha}..{source_sha}",
                        )
                    )
                    self.assertEqual(ahead, expected)
                    api = MemoryGitHub(ahead, source_sha, target_sha)
                    result = sync.synchronize(api, REPOSITORY)
                    self.assertEqual(result["source"], source_sha)
                    self.assertEqual(result["target"], target_sha)
                    self.assertEqual(result["ahead"], expected)
                    if expected == 0:
                        self.assertEqual(result["status"], "up_to_date")
                        self.assertEqual(api.writes, [])
                    else:
                        self.assertEqual(result["status"], "created")
                        self.assertEqual(len(api.writes), 1)

    def test_creation_is_idempotent_and_uses_a_manual_merge_pr(self):
        api = MemoryGitHub(ahead=4)
        created = sync.synchronize(api, REPOSITORY)
        self.assertEqual(created["status"], "created")
        self.assertEqual(len(api.pulls), 1)
        body = api.pulls[0]["body"]
        self.assertTrue(body.startswith(sync.MARKER))
        self.assertIn(SOURCE_SHA, body)
        self.assertIn(TARGET_SHA, body)
        self.assertIn("**4**", body)
        self.assertIn("merged manually", body)
        self.assertIn("separate branch", body)
        self.assertIn("Approve workflows", body)
        again = sync.synchronize(api, REPOSITORY)
        self.assertEqual(again["status"], "unchanged")
        self.assertEqual(again["url"], created["url"])
        self.assertEqual(len(api.writes), 1)

    def test_existing_owned_pr_is_updated_when_either_tip_changes(self):
        api = MemoryGitHub()
        sync.synchronize(api, REPOSITORY)
        api.source = "c" * 40
        api.target = "d" * 40
        api.ahead = 2
        result = sync.synchronize(api, REPOSITORY)
        self.assertEqual(result["status"], "updated")
        self.assertEqual(len(api.pulls), 1)
        self.assertEqual(api.writes[-1][0], "PATCH")
        self.assertEqual(api.pulls[0]["title"], sync.TITLE)
        self.assertIn(api.source, api.pulls[0]["body"])
        self.assertIn(api.target, api.pulls[0]["body"])
        self.assertNotIn(SOURCE_SHA, api.pulls[0]["body"])

    def test_existing_human_pr_is_preserved_and_not_duplicated(self):
        for body in [None, "Human notes", "Notes before " + sync.MARKER]:
            with self.subTest(body=body):
                api = MemoryGitHub()
                existing = pull(body=body)
                api.pulls.append(copy.deepcopy(existing))
                result = sync.synchronize(api, REPOSITORY)
                self.assertEqual(result["status"], "existing_manual_pr")
                self.assertEqual(api.pulls, [existing])
                self.assertEqual(api.writes, [])

    def test_exact_head_repository_and_branch_pair_are_required(self):
        api = MemoryGitHub()
        foreign = pull(repository="somebody/FlagGems-sglang")
        wrong_head = pull(number=13)
        wrong_head["head"]["ref"] = "unrelated"
        wrong_base = pull(number=14)
        wrong_base["base"]["ref"] = "unrelated"
        api.pulls = [foreign, wrong_head, wrong_base]
        self.assertEqual(
            sync.synchronize(api, REPOSITORY)["status"], "created"
        )
        self.assertEqual(len(api.writes), 1)
        api = MemoryGitHub()
        api.pulls = [pull(repository=REPOSITORY.upper())]
        self.assertEqual(
            sync.synchronize(api, REPOSITORY)["status"], "existing_manual_pr"
        )
        self.assertEqual(api.writes, [])

    def test_duplicate_exact_pair_fails_without_mutation(self):
        api = MemoryGitHub()
        api.pulls = [pull(), pull(number=13)]
        with self.assertRaisesRegex(RuntimeError, "Multiple open sync PRs"):
            sync.synchronize(api, REPOSITORY)
        self.assertEqual(api.writes, [])

    def test_creation_race_recovers_the_other_exact_pair_pr(self):
        for owned in (False, True):
            with self.subTest(owned=owned):
                api = MemoryGitHub()
                api.creation_error = sync.GitHubError("already exists", 422)
                body = (
                    sync.MARKER + " old snapshot" if owned else "Human notes"
                )
                api.concurrent_pull = pull(number=44, body=body)
                result = sync.synchronize(api, REPOSITORY)
                self.assertEqual(
                    result["status"],
                    "updated" if owned else "existing_manual_pr",
                )
                self.assertTrue(result["url"].endswith("/44"))
                self.assertEqual(len(api.pulls), 1)
                self.assertEqual(len(api.writes), 2 if owned else 1)

    def test_other_creation_errors_and_unexplained_422_are_not_hidden(self):
        for status in (401, 403, 422, 500):
            with self.subTest(status=status):
                api = MemoryGitHub()
                error = sync.GitHubError("request denied", status)
                api.creation_error = error
                with self.assertRaises(sync.GitHubError) as raised:
                    sync.synchronize(api, REPOSITORY)
                self.assertIs(raised.exception, error)

    def test_dry_run_does_not_create_or_update(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                api = MemoryGitHub()
                if existing:
                    api.pulls = [pull(body=sync.MARKER + " old")]
                before = copy.deepcopy(api.pulls)
                result = sync.synchronize(api, REPOSITORY, dry_run=True)
                self.assertEqual(
                    result["status"],
                    "would_update" if existing else "would_create",
                )
                self.assertEqual(api.pulls, before)
                self.assertEqual(api.writes, [])

    def test_invalid_repository_and_comparison_fail_before_mutation(self):
        for repository in (
            "missing-owner",
            "a/b/c",
            "a/b?token=x",
            "a b/repo",
        ):
            with self.subTest(repository=repository):
                api = MemoryGitHub()
                with self.assertRaisesRegex(ValueError, "OWNER/REPOSITORY"):
                    sync.synchronize(api, repository)
                self.assertEqual(api.calls, [])
        for ahead in (True, -1, "2", 1.5, None):
            with self.subTest(ahead=ahead):
                api = MemoryGitHub(ahead=ahead)
                with self.assertRaisesRegex(ValueError, "invalid ahead_by"):
                    sync.synchronize(api, REPOSITORY)
                self.assertEqual(api.writes, [])


class GitHubWrapperAndCliTests(unittest.TestCase):
    def test_gh_uses_environment_authentication_and_json_stdin(self):
        fake_token = "fictional-token-not-for-command-arguments"
        completed = subprocess.CompletedProcess([], 0, '{"ok": true}', "")
        with patch.dict(os.environ, {"GH_TOKEN": fake_token}):
            with patch.object(
                sync.subprocess, "run", return_value=completed
            ) as run:
                result = sync.github_request(
                    "POST", API_ROOT + "/pulls", {"body": "a\nb"}
                )
        self.assertEqual(result, {"ok": True})
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
                "gh",
                "api",
                "--method",
                "POST",
                API_ROOT + "/pulls",
                "--input",
                "-",
            ],
        )
        self.assertEqual(
            json.loads(run.call_args.kwargs["input"]), {"body": "a\nb"}
        )
        self.assertNotIn(fake_token, repr(run.call_args))
        with patch.object(
            sync.subprocess, "run", return_value=completed
        ) as run:
            sync.github_request("GET", API_ROOT + "/branches/master")
        self.assertIsNone(run.call_args.kwargs["input"])
        self.assertNotIn("--input", run.call_args.args[0])

    def test_gh_error_retains_status_and_explains_pr_permission_failure(self):
        for status in (403, 422, 500):
            with self.subTest(status=status):
                completed = subprocess.CompletedProcess(
                    [], 1, "", f"gh: denied (HTTP {status})"
                )
                with patch.object(
                    sync.subprocess, "run", return_value=completed
                ):
                    with self.assertRaises(sync.GitHubError) as raised:
                        sync.github_request("POST", API_ROOT + "/pulls", {})
                self.assertEqual(raised.exception.status, status)
                if status == 403:
                    self.assertIn(
                        "pull-requests: write", str(raised.exception)
                    )
                    self.assertIn(
                        "Allow GitHub Actions", str(raised.exception)
                    )

    def test_cli_reports_result_and_adds_manual_merge_summary(self):
        api = MemoryGitHub()
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            stdout = io.StringIO()
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
                with patch.object(
                    sync, "github_request", api
                ), redirect_stdout(stdout):
                    code = sync.main(["--repo", REPOSITORY, "--dry-run"])
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["status"], "would_create"
            )
            self.assertIn("Manual merge only", summary.read_text())
            self.assertIn("would_create", summary.read_text())
            self.assertEqual(api.writes, [])

    def test_cli_uses_environment_repository_and_errors_are_nonzero(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(
            os.environ, {"GITHUB_REPOSITORY": REPOSITORY}, clear=True
        ):
            with patch.object(sync, "github_request", MemoryGitHub(ahead=0)):
                with redirect_stdout(stdout):
                    self.assertEqual(sync.main([]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "up_to_date")
        with patch.object(
            sync,
            "github_request",
            side_effect=sync.GitHubError("permission denied", 403),
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(sync.main(["--repo", REPOSITORY]), 1)
        self.assertIn("Branch sync failed", stderr.getvalue())
        self.assertIn("permission denied", stderr.getvalue())

    def test_cli_missing_repository_is_an_explicit_usage_error(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(
            io.StringIO()
        ) as stderr:
            with self.assertRaises(SystemExit) as raised:
                sync.main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("provide --repo", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
