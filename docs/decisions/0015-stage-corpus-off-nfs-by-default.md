# ADR 0015 -- the harness stages the corpus to local disk by default

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user ("understand why this happens and make sure it never happens again")

## Context

Two multi-hour wedges in one evening (`ladder15k` wave 2, `rho_held_trust`):
tasks sitting at 0% GPU utilisation, memory allocated, `buff/cache` stuck at
14-18 GB against a ~20 GB corpus with ~970 GB of RAM free. Root cause found
and written up in full in `docs/CLUSTER.md` ("The root cause, found"):
`$DATA_p330` is `hard`-mounted NFS4.2 from one server, and the `cachefilesd`
daemon backing its `fsc` (FS-Cache) mount option has been dead --
`SIGSEGV`, core-dumped -- since 2026-09-08, a root-owned service this project
cannot restart. `hard` means a slow server hangs the client's read rather than
erroring it, and the training loop's scattered per-step reads (`TokenCorpus`
samples random windows across the whole corpus) are exactly the pattern that
triggers it.

## Decision

`harness/local_cache.py::stage_locally` copies `data_path`/`val_path` to local
disk (`RunConfig.local_cache_dir`, default `/tmp/c4`) before `TokenCorpus`
opens them, for any file at least 500 MB (below that, staging is pure
overhead: `c4_val.bin` is 16 MB and is read directly). This runs automatically
inside `harness/train.py::train` -- **no sbatch script has to remember to call
anything**, which is the point: a rule that depends on every future script
carrying a line is a rule that eventually gets forgotten, the way `ngd_power`
and `pion_ablated`'s manifest bug happened here before. `local_cache_dir = ""`
turns it off. `scripts/stage_corpus.sh` is the same logic as a standalone
shell script, for the (currently nonexistent) case of something that reads the
corpus without going through the harness.

Deployed regardless of whether or when `cachefilesd` gets fixed: even a
healthy NFS server is one more moving part than a local disk for the single
hottest, most-reread, entirely static file this project touches, and the copy
costs 32 seconds once per node.

## Consequences

`data_path`/`val_path` in the manifest still record the canonical
`$DATA_p330` path (unaffected: staging happens after `RunConfig` is built, and
`local_cache_dir` was added to `_EXCLUDED` so it does not move the config
hash) -- a manifest is not misleading about where the canonical data lives,
only the harness's own process reads from a local copy internally.

Requires ~15 GB of free space in `local_cache_dir` on whichever node runs a
job; `/tmp` measured at 424 GB free, so this is not tight, but a node with a
much smaller or full `/tmp` would fall back to the original NFS path rather
than fail the run (`stage_locally` never raises).

**Not covered:** run directories (`$DATA_p330/runs`) stay on NFS. Small,
infrequent traffic relative to the corpus reads, and not what triggered
tonight's wedges, so left alone; revisit if a wedge recurs with the corpus
already local.

**Still open, and not this project's to fix:** `cachefilesd` itself. Report to
cluster admins with the evidence in `docs/CLUSTER.md` -- likely a one-command
fix for someone with root, invisible until `systemctl status cachefilesd` was
actually checked.

## Evidence

`docs/CLUSTER.md`, "The root cause, found" (mount options, `nfsstat -c`,
`systemctl status cachefilesd`, `/tmp`'s `df -hT` and `findmnt` output showing
it as local ZFS). `tests/test_local_cache.py`, 7 tests, pins: below-threshold
files are not staged, `local_cache_dir = ""` disables it, a large file is
copied once and reused (a second call does not re-copy), a wrong-size local
copy is replaced rather than trusted, a missing source is left for the real
`open()` to report, an unwritable cache directory falls back to the original
path rather than raising, and four concurrent callers racing to stage the same
file produce one copy, not four, and never leave a `.partial` behind. Full
suite: 268 passed, 1 skipped (was 261 before this). End-to-end CPU smoke test:
`harness.train.train()` against the real `$DATA_p330/c4/c4_train.bin` path
with `local_cache_dir="/tmp/c4"` ran 3 steps successfully, reading from the
staged local copy.
