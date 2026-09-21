# ADR 0003 -- one spectral floor, max(λ, ε·λ_max), with ε = 1e-4

- **Status:** Accepted
- **Date:** 2026-08-25. Recorded retroactively on 2026-09-22 from `AGENTS.md`, `ALGORITHM.md` and the journal; the original decision is older, and the journal is the source for what was measured at the time.
- **Decided by:** project

## Context

Three spectra can degenerate (`A`, `WᵀW`, `W A Wᵀ`), and the dead block of a
non-square weight is a genuine `0/0`.

## Decision

A single relative floor `λ ← max(λ, ε·λ_max)` on every spectrum that reaches a
denominator, the pencil's own output included. `ε = 1e-4`. There is one knob, not
one per spectrum. No shift, no mask, no truncation.

## Consequences

`ε` is a safety device, not a hyperparameter. Its lower bound is set by the
compute dtype: fp32 machine epsilon is 1.2e-7, so `1e-8` is meaningless there.

## Evidence

On a wide layer the floor is **134x** more accurate than a shift at `ε = 1e-4`.
The plateau is four orders wide in fp64; in fp32 `ε = 1e-8` gives a relative
error of 2.2e-1 against fp64. The floor was later checked and found not to be
eating the Fisher: `iso-cos` 0.82 at the median with 3443x of dynamic range
(journal, 2026-08-29).
