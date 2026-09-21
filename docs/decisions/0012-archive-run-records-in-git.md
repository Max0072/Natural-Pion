# ADR 0012 -- archive the small run records in git?

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user

## Context

`runs/` is in `.gitignore` and lives on `$DATA_p330`, outside the repository:
181 GB across 323 runs. Of that, 205 GB is checkpoints and 530 MB is
`diagnostics.jsonl`, but only **13 MB** is `manifest.json` and `log.jsonl`, the
configuration and the loss curve of every run, which is where every number in
`docs/` comes from. The user asked that everything be on git so nothing is lost.

## Options considered

1. **Commit `manifest.json` and `log.jsonl` of every run** under `runs-record/`
   (13 MB), leaving checkpoints and diagnostics out. Any figure in the docs can
   then be re-checked from GitHub.
2. Leave `runs/` as it is, on disk only.

## Decision

Option 1. `manifest.json` and `log.jsonl` of every run are copied to
`runs-record/<investigation>/<run>/` by `scripts/archive_runs.py` and committed;
`runs/README.md` comes along as `runs-record/RUNS_README.md`. First sync on
2026-09-22: 646 files, 13.4 MB, 323 runs, none missing.

## Consequences

Checkpoints and `diagnostics.jsonl` stay on disk only; the checkpoints
regenerate by re-running, the diagnostics do not and are not backed up by this.
The archive is a snapshot: it must be refreshed by re-running the script after
jobs finish (a log copied mid-run is a prefix of the final one). **The six
stage-1 runs of job 332045 were still in flight at the first sync**, so the
archive holds prefixes of them until it is re-run.
