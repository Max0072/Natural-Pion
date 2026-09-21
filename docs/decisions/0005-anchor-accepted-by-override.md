# ADR 0005 -- the harness anchor is accepted although it missed its own criteria

- **Status:** Accepted (by override)
- **Date:** 2026-08-26
- **Decided by:** the user

## Context

The anchor runs Pion's own configuration in this harness and checks it against
their published 60M figures (3.3575 bilateral, 3.3654 alternate). Criteria were
written down before the numbers: gap within 0.005 of 0.0079 and `matched: true`
on both levels. The fallback was to stop looking for a fifth difference and run
their unmodified code as a control.

## Decision

Round 3 (bilateral 3.3805, alternate 3.3954, arm gap 0.0149) meets neither
criterion. The user overrode the fallback and accepted the anchor: the level
reproduces to under 1% (0.69% and 0.89%) and the ordering and gap are
qualitatively right.

## Consequences

It licenses comparisons *inside this harness*. It does **not** license quoting
our Pion number as a reproduction of theirs, or any claim resting on the
magnitude of the bilateral/alternate gap, which is 1.9x theirs. The paper states
this as a limitation, in the text. The unmodified-code control is deprioritised,
not cancelled.

## Evidence

Three rounds: 3.3997/3.4352 (gap 0.0355), 3.4021/3.4161 (0.0140), 3.3805/3.3954
(0.0149). The last two fixes (AdamW betas, no decay on 1-D parameters) halved the
level error and left the gap untouched. Journal 2026-08-26.
