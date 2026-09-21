# Run records

A copy of the small record of every run in `$DATA_p330/runs/` (323 runs at the
time of writing), so that every number in `docs/` can be re-read from this
repository. Decision: `docs/decisions/0012-archive-run-records-in-git.md`.

Layout: `<investigation>/<run>/{manifest.json, log.jsonl}`, the same paths as
under `runs/`. `RUNS_README.md` is `runs/README.md` verbatim -- it says what each
investigation directory is for.

* `manifest.json` -- the full `RunConfig`, its hash (which names the directory),
  the git commit the run started from, tokens per step and the machine. **It
  records the configuration, not always the object built from it**: the removed
  `pion_ablated` runs in `ablated/` record the un-ablated settings.
* `log.jsonl` -- one JSON row per logged step (`train_loss`, learning rates,
  diagnostics), `val_loss` rows at each evaluation, and an `event: start` marker
  per attempt (a resumed run appends; take the last one).

**Not here, and still only on disk under `$DATA_p330/runs/`:** the checkpoints
(`checkpoint.pt`, 205 GB, they regenerate by re-running) and
`diagnostics.jsonl` (530 MB of per-layer optimizer diagnostics).

To refresh after a job ends:

    python scripts/archive_runs.py            # copies what changed, skips the rest
    git add runs-record && git commit

Runs that were still in flight when this was written are listed by the script; a
log copied mid-run is a prefix of the final one, so re-run it when they finish.
