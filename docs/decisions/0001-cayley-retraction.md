# ADR 0001 -- NGD-Pion retracts with Cayley, not Pion's truncated exponential

- **Status:** Accepted
- **Date:** 2026-08-24/25 (design entries of the journal). Recorded retroactively on 2026-09-22 from `AGENTS.md`, `ALGORITHM.md` and the journal; the original decision is older, and the journal is the source for what was measured at the time.
- **Decided by:** project

## Context

Pion retracts a step onto the rotation group with a degree-2 truncated
exponential, `R = I + Ω + Ω²/2` for a skew step `Ω`. Then `RᵀR = I + Ω⁴/4`
exactly, so the truncation always inflates and the singular values of `W` drift.

## Decision

Every NGD-Pion variant retracts with the Cayley map,
`Cayley(-cX) = (I + c/2·X)⁻¹ (I − c/2·X)`, which is exactly orthogonal for skew
`X` at any step size. The published-Pion arm keeps its own truncated exponential:
it is the baseline and is not modified (ADR 0008).

## Consequences

The spectrum of every rotated weight is preserved exactly, which is the
property the method is built on.

## Evidence

Relative singular-value drift over 2000 steps at angle 0.05: **7.1e-15** for
Cayley, **2.4e-4** for the truncated exponential (`ALGORITHM.md`, "Осознанные
отличия от бейзлайна"). At angle 5 the truncated exponential's spectral norm is
12.5.

The argument that Cayley is "a precondition of the ablation" no longer stands:
the ablated arm it belonged to was withdrawn (ADR 0008).
