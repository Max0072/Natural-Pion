# ADR 0002 -- optimizer linear algebra in true fp32, TF32 off around it

- **Status:** Accepted
- **Date:** 2026-08-24/25. Recorded retroactively on 2026-09-22 from `AGENTS.md`, `ALGORITHM.md` and the journal; the original decision is older, and the journal is the source for what was measured at the time.
- **Decided by:** project

## Context

On Ampere and newer, torch does fp32 matrix products in TF32 (ten mantissa
bits) by default. The CPU test suite cannot see this: TF32 does not exist there.

## Decision

The retraction, the factorisation and the covariance accumulation run in fp32
with TF32 switched off around them (`exact_fp32`), and the covariance `A` is
stored in fp32 at minimum. The model's own forward and backward keep TF32 and
bf16. fp64 was rejected.

## Consequences

`scripts/gpu_smoke.py` is the only check that can see a regression, and needs a
card. It was not re-run in the 2026-09-22 audit.

## Evidence

Measured on an RTX PRO 6000 Blackwell: Cayley orthogonality error **4.3e-3**
with TF32 against **3.9e-6** without; relative drift of the singular values over
200 two-sided steps **1.0** against **2.6e-4**. TF32 on the model is worth
2.2-2.6x at a relative 1e-3 cost on a gradient that is about half noise. fp64
buys nothing (end-to-end error 1e-5 to 1e-3 on real spectra) and costs 30-60x on
consumer GPUs. bf16 *activations* perturb `A` by 4.5e-5, bf16 *storage* by
7.4e-3, which sits 70x above the floor in ADR 0003.
