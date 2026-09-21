# ADR 0007 -- short runs do not rank these optimizers; every arm gets its own learning rates

- **Status:** Accepted
- **Date:** 2026-08-27/28
- **Decided by:** project (measurement)

## Context

Every sweep before 2026-08-27 used 150 steps. Measured on 2026-08-27, that
horizon does not merely fail to predict the full-length order, it inverts it.

## Decision

1. A conclusion about which optimizer is better is not drawn from 150 steps, and
   3000 steps is treated as directional only.
2. **Never compare two configurations at a shared learning rate**, and never
   compare a grid edge against another arm's optimum.
3. Quote a step-equivalent speed-up, not a loss difference: loss is compressive,
   so a constant step-equivalent advantage shows up as a shrinking gap.

## Consequences

Most "dead end" verdicts in `ngd_pion/README.md` were decided at 150 steps and
mean "lost a comparison we no longer trust". Hence the horizon ladder in
`docs/PLAN.md` (H4).

## Evidence

`pion` beats `ngd-pion` by 0.30 at 73 242 steps and loses by 0.24 at 150. For
`shampoo-pion`, `eta = 1e-1` is worst at step 500 and best from step 1000. For
`ngd-pion-m`, `2e-2` wins at 3000 steps and loses at length. Four orderings in
all have flipped between a short protocol and the full length.
