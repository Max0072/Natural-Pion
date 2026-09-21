# ADR 0006 -- the live NGD-Pion arm measures the backward covariance S (S = I reversed)

- **Status:** Accepted; supersedes the `S = I` decision
- **Date:** 2026-08-27
- **Decided by:** project (measurement)

## Context

`S = I` had been decided on a toy transformer with about 80k tokens estimating
`S` at `d_out = 256`, and the row below it in `AGENTS.md` said not to put the
claim in the paper until it was checked at scale. The measured `S` had never been
given its own learning rate.

## Decision

`ngd-pion-s`, which uses the measured `S`, is the live arm and the one compared
against Pion. `S = I` (`ngd-pion`) is superseded.

## Consequences

`ALGORITHM.md` still states `S = I` in its decision tables and was not updated;
recorded as documentation debt in `docs/VALIDATION.md` V6.

## Evidence

Job 273026 swept the measured `S` at **its own** `eta`, which is a hundred times
smaller than the one used before: it wins by **0.23-0.32** against a 0.07 floor.
