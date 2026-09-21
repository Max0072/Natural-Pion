# ADR 0012 -- archive the small run records in git?

- **Status:** Proposed (waiting on the user)
- **Date:** 2026-09-22
- **Decided by:** --

## Context

`runs/` is in `.gitignore` and lives on `$DATA_p330`, outside the repository:
181 GB across 323 runs. Of that, 205 GB is checkpoints and 530 MB is
`diagnostics.jsonl`, but only **13 MB** is `manifest.json` and `log.jsonl`, the
configuration and the loss curve of every run, which is where every number in
`docs/` comes from. The user asked that everything be on git so nothing is lost.

## Options

1. **Commit `manifest.json` and `log.jsonl` of every run** under `runs-record/`
   (13 MB), leaving checkpoints and diagnostics out. Any figure in the docs can
   then be re-checked from GitHub.
2. Leave `runs/` as it is, on disk only.

## Decision

Not taken.
