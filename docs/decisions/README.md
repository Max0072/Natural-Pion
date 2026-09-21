# Decision records

One file per decision. `docs/JOURNAL.md` is the chronological story of how the
project got here; this directory is the **list of what was decided**, with the
reason and the evidence, so that nobody has to read 5500 lines to learn why
something is the way it is. `AGENTS.md` and `docs/RESUME.md` link here rather than
restate.

**Rules**

* One decision per file, numbered in the order it was recorded: `NNNN-short-name.md`,
  from `0000-template.md`.
* Do not rewrite an accepted record's decision. If the decision changes, write a
  new record, mark the old one `Superseded by ADR NNNN` and say what changed and
  why. The old text stays: what was believed, and on what evidence, is the point.
  (Facts inside a record may be corrected with a dated note.)
* Every record carries its evidence. A decision with no measurement beside it
  gets reopened.
* `Proposed` means a question is open and the user has not answered it. Nothing
  proposed is acted on.
* Records 0001-0007 were written on 2026-09-22 from what the repository already
  said; they say so, and their dates are the original decisions'.
* Add the record to the table below in the same commit.

| ADR | decision | status | date |
|---|---|---|---|
| [0001](0001-cayley-retraction.md) | NGD-Pion retracts with Cayley, not the truncated exponential | Accepted | 2026-08-24/25 |
| [0002](0002-fp32-with-tf32-off.md) | Optimizer linear algebra in true fp32, TF32 off around it | Accepted | 2026-08-24/25 |
| [0003](0003-one-spectral-floor.md) | One spectral floor, max(λ, ε·λ_max), ε = 1e-4 | Accepted | 2026-08-25 |
| [0004](0004-same-corpus-and-budget.md) | The corpus is their corpus; the budget is 9.6B tokens | Accepted | 2026-08-25 |
| [0005](0005-anchor-accepted-by-override.md) | The harness anchor is accepted although it missed its criteria | Accepted (override) | 2026-08-26 |
| [0006](0006-measured-backward-covariance.md) | The live arm measures the backward covariance S (S = I reversed) | Accepted | 2026-08-27 |
| [0007](0007-short-protocols-do-not-rank.md) | Short runs do not rank these optimizers; every arm gets its own rates | Accepted | 2026-08-27/28 |
| [0008](0008-baseline-is-published-pion.md) | The baseline is published Pion; there is no ablated arm | Accepted | 2026-09-22 |
| [0009](0009-classical-head-to-head-protocol.md) | How NGD-Pion is compared with Pion: classical batch, both tuned, seed-paired | Accepted | 2026-09-22 |
| [0010](0010-stage-1-seeds-run-concurrently.md) | Stage 1 runs all three seeds at once on one node | Accepted | 2026-09-22 |
| [0011](0011-delete-pion-ablated-code.md) | Delete `pion_ablated` from the code | Accepted | 2026-09-22 |
| [0012](0012-archive-run-records-in-git.md) | Archive the small run records in git | Accepted | 2026-09-22 |
