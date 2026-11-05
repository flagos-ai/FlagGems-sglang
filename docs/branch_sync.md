# Branch synchronization

`master` contains shared operator implementations and infrastructure.
`kernelgen-dev` receives new operator correctness tests, benchmarks and
Definitions. Changes flow from `master` into `kernelgen-dev` through a reviewed
pull request; the reverse direction is handled through individual feature PRs.

## Daily check

The **Sync master to kernelgen-dev** workflow runs daily at **04:00
Asia/Shanghai (20:00 UTC on the preceding day)**. GitHub schedules can be
delayed; this is a daily check, not an exact-time guarantee. It also supports
**Run workflow** on `master`, with a read-only `dry_run` option.

The workflow must first be merged into the default branch, `master`. It is not
active while its implementation PR is open, and dispatches from other branches
or forks are deliberately skipped.

The checker compares commit ancestry, rather than file differences. If all
`master` commits are already in `kernelgen-dev`, it does nothing. Otherwise it
creates or reuses the single PR with **head `master`, base `kernelgen-dev`**.
GitHub keeps that PR's code comparison current as either branch advances. The
daily checker updates its generated description only when the checked commits
change. An existing human-written description is preserved.

There is no extra sync branch, Git push, forced update or automatic merge. No
operator tests, benchmarks or Definitions are changed by the checker. A failed
API request fails the workflow visibly; a concurrent PR creation is recovered
without opening a duplicate.

## Review and merge

1. Review the sync PR and its CI checks. `basic ci` covers both `master` and
   `kernelgen-dev` pushes and PR targets. The existing benchmark step captures
   errors in its output without failing the whole job, so inspect the benchmark
   logs as well as the overall check status.
2. If checks are waiting for approval, select **Approve workflows to run**.
3. Merge manually with **Create a merge commit**. Do not squash/rebase this
   long-lived branch sync: preserving `master` ancestry lets the next check
   correctly identify which commits still need synchronization.

Conflicts stay visible on the PR and block merging. Do **not** use this PR's
**Update branch** button or web conflict editor: the head is `master`, so those
would put development-side changes onto `master`. Instead, create a temporary
repair branch from `kernelgen-dev`, merge `master` there, resolve conflicts and
submit the result as a PR to `kernelgen-dev`. The next daily check will then
see the source commits as integrated. The automation never chooses one side
of a conflict.

## Authentication

The workflow uses the built-in `GITHUB_TOKEN`, with only `contents: read` and
`pull-requests: write`. No personal token or new repository secret is required.
Repository/organization policy must allow Actions to create PRs: **Settings →
Actions → General → Workflow permissions → Allow GitHub Actions to create and
approve pull requests**. This grants the capability; this workflow does not
approve or merge anything. A permission error reports that setting in the job
log. An administrator may need to enable it if organization policy disables it.

GitHub can put CI runs for PRs created/updated by `GITHUB_TOKEN` into an
approval-required state. This is separate from reviewing and merging the PR.
No `ops-test` run is dispatched automatically, so synchronization does not
publish monitoring results or send Feishu messages.

## Local verification

```sh
python -m unittest discover -s tools/tests -p 'test_sync_kernelgen_dev.py' -v
python tools/sync_kernelgen_dev.py --repo flagos-ai/FlagGems-sglang --dry-run
```

The first command uses no device runtime or GitHub credentials. The second
uses the local `gh` authentication and only reads GitHub state.

References: [GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
[workflow token and CI triggering](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#triggering-a-workflow-from-a-workflow).
