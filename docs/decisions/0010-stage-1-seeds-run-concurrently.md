# ADR 0010 -- stage 1 runs all three seeds at once on one node

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user

## Context

The project rule (memory of 2026-08-27, measured 18.7 s/step against 4.4 s/step)
is that concurrent runs on one node should share a seed, or they read different
corpus windows and starve each other for I/O. The first version of the stage-1
job ran the seeds one after another (`%1`).

## Decision

The user said it is not necessary and to run everything in parallel. The array
throttle of job 332045 was raised from 1 to 3 in place, so six runs (three seeds x
two arms) share `rtx6002`.

## Consequences

A cost is possible and was checked instead of assumed: at twelve minutes
`ngd-pion-s` ran at about 0.87 s/step (solo 0.967) and `pion` at about 0.54
(solo 0.46). No contention, on a node whose corpus was already resident. One node
in one cache state; if a later seed batch is several times slower than solo,
suspect this first. This does not repeal the rule.

## Evidence

Job 332045, `scripts/sbatch/seeds_head.sbatch`, output in
`$DATA_p330/runs/seedshead/`. Journal 2026-09-22, addendum of 01:07.
