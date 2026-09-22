# ADR 0014 -- `adapt_damping` steers on `rho_held` when the holdout is measured

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user, following the K-FAC/Levenberg-Marquardt discussion

## Context

`alpha = quad/curv` (§6, `ALGORITHM.md`) is not a trust region: both quantities
are computed from the model's own internal, possibly stale linear approximation,
never from the realised loss, so it cannot detect that a step was too long --
only that the eigenbasis has drifted (journal, 2026-09-22, the `trust_region_alpha`
audit). A genuine trust-region / Levenberg-Marquardt signal already exists in
this codebase: `rho`, the realised loss reduction divided by the model's
predicted reduction. It was tried twice and lost both times --
`ngd_trust_lr` (adapts `eta` on `rho`) settled 2-4x above the swept optimum
(job 297936/298962), and `ngd-pion-damped`'s Levenberg-Marquardt rule on a
damping term was worth +0.39-0.40 by itself but its carrier also removed the
spectral floor, costing 0.72 net (job 297755). Both used the *same-batch* `rho`,
which the 2026-08-27 journal entry already named as the likely reason the
`ngd_trust_lr` controller settles too high: it is evaluated on the very batch
whose gradient produced the step, so it cannot see that the step is too long
when the step is mostly sampling noise (`split_half_step.py`, 96.4%).
`rho_held`, measured on an independent batch, has existed since 2026-08-29
(`rho_holdout`) but was deliberately left unwired -- "diagnostic only... one
change at a time."

## Decision

`harness/train.py`: `adapt_damping` is called with `rho_held` when it was
measured (`rho_holdout` on), and with same-batch `rho` otherwise. This is the
one change the 2026-08-29 comment deferred. `RunConfig.rho_holdout`'s docstring
is updated to say so -- turning the holdout on now changes the trajectory of any
run with `ngd_trust_lr` on, not only what gets logged.

Also recorded in passing, found while reading the code for this: the defaults
`ngd_trust_lr_lo = 0.25`, `ngd_trust_lr_hi = 0.75` were never updated to match
the comment beside them, which already argues for `[0.8, 1.2]` (job 298962's
corrected band). Left as is; flagged in the docstring; pass the two flags
explicitly until someone changes the default deliberately.

## Consequences

`ngd-pion-s` with `--ngd-trust-lr true --rho-holdout true` now runs a genuine,
un-replaced Levenberg-Marquardt-style step-size correction on top of the
existing spectral floor -- the combination the 2026-08-28 journal named as
untested. No `RunConfig` field changed, so run hashes for every existing
configuration (holdout off, or trust_lr off) are unchanged; only the trajectory
of the specific combination `rho_holdout=true, ngd_trust_lr=true` changes,
and nothing used that combination before today (checked: `fisher.sbatch` sets
`rho_holdout true` with `ngd-trust-lr false`).

Genuinely uncertain whether this wins. `rho_held` is itself a single noisy
realised-loss comparison, and the step it is meant to correct is 96.4% sampling
noise; the same noise that limits the raw method may limit this correction too.
Stated as a prediction to check, not a claim.

## Evidence

CPU smoke test (12 steps, tiny random corpus, `rho_every=1`, `rho_holdout=true`,
`ngd_trust_lr=true`): `rho_held` present at every logged step, `lr_scale`
moves in response (0.667 -> 0.012 over 12 steps on this toy input) -- wiring
confirmed end to end. 261 tests still pass, 1 skipped (`adapt_damping`'s own
unit tests call it directly with explicit `rho` values and do not exercise this
call site). Job 333063, `scripts/sbatch/rho_held_trust.sbatch`: four runs,
`ngd-pion-s`, adamw 8e-3, rot in {3e-3, 6e-3} x {same-batch rho, held rho},
B=512, 3000 steps, seed 0, band `[0.8, 1.2]`, `rho_every 5`, `rho_micro 128`
(the combination `fisher.sbatch` already ran without an OOM). Submitted
2026-09-22.
