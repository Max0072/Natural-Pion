# ADR 0008 -- the baseline is published Pion; there is no ablated arm

- **Status:** Accepted; supersedes the `pion_ablated` comparison design
- **Date:** 2026-09-22
- **Decided by:** the user

## Context

The comparison design of `ALGORITHM.md` and `README.md` measured NGD-Pion
against `pion_ablated` (Pion with momentum and RMS scaling switched off and the
retraction forced to Cayley), so that one variable separated the arms. Stage 2 of
the plan would have tuned that arm.

## Decision

The baseline is **published Pion, and only that**. NGD-Pion is compared to it as
a whole method. No claim is made about which component of NGD-Pion is
responsible for a gain. Hypothesis H3 and stage 2 of `docs/PLAN.md` are
withdrawn. User's words: an ablated Pion "should not work, because the algorithm
only works whole".

## Consequences

The paper can say NGD-Pion beats Pion, if the evidence bears it out. It cannot say
the Fisher preconditioner is the reason. The audit ledger row for that claim reads
"not claimed". The code for `pion_ablated` is untouched pending ADR 0011.

## Evidence

Without RMS scaling Pion's own retraction diverges within tens of steps, so the
ablated arm must borrow Cayley and is a different optimizer. At 150 steps it is
0.40 worse than freezing the matrices (val 6.109 against 5.7071), and no run of it
exists beyond 150 steps. Its manifests record the un-ablated settings. And the
plan could not have separated "the preconditioner helps" from "the preconditioner
replaces the normalisation Pion has for free": that needs an NGD-plus-RMS arm that
does not exist. Journal 2026-09-22 (second entry).
