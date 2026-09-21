# ADR 0009 -- how NGD-Pion is compared with Pion: classical batch, both tuned, seed-paired

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user, with the audit's findings

## Context

The 2026-08-29 headline ("a tie, second of nine, inside Pion's own spread") was a
comparison against eight runs that are not replicates (two are the alternate
variant, six predate two harness fixes worth about 0.045). Like-for-like, at full
length with AdamW pinned, NGD-Pion is 0.014 behind, one run each.

## Decision

- **Setting:** B = 512 sequences of 256 tokens (131 072 tokens per step), the
  published Pion 60M setting.
- **Arms:** `ngd-pion-s` against `pion`, each with its own tuned `rot`
  (rotational learning rate) and `adamw` (AdamW learning rate on the 56.5% of the
  model Pion does not own). `ngd-pion-s` and not `ngd-pion-m`: momentum is ahead at
  3000 steps and loses at full length (3.3881 and 3.3941 against 3.3860).
- **Pairing:** same seed for both arms, so both see the same initialisation and the
  same data order. Diff the manifests before interpreting any pair.
- **Rule for the short-run claim (H1), fixed before the numbers:** four paired
  gaps (seed 0 plus three), all positive, mean gap / standard error > 2.35
  (one-sided 5%, 3 degrees of freedom).
- **Order:** an error bar at 3000 steps, then a 15 000-step horizon ladder, then
  the full length, reported as one tuned run against one tuned run, n = 1 each.

## Consequences

Numbers from earlier runs with AdamW pinned at 1e-3 are context, not the
comparison. A set of runs is not a set of replicates until the manifests say so.

## Evidence

`docs/VALIDATION.md` V1-V4 and `docs/PLAN.md`. At B = 512, 3000 steps, seed 0:
3.7954 against 3.8412, gap +0.046, no error bar yet.
