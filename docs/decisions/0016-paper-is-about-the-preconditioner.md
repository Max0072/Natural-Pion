# ADR 0016 -- the paper is about the preconditioner, not about beating Pion outright

- **Status:** Accepted
- **Date:** 2026-09-23
- **Decided by:** the user

## Context

The seeded, full-length evaluation pipeline this project's own decision record
(`aim-for-a-positive-result-paper` memory, and the parallel push behind
ADR 0009's protocol) was built to support found a clear, if unwelcome, answer:

* **H1 holds.** At B = 512, 3000 steps, both optimizers tuned, `ngd-pion-s`
  beats published Pion by +0.035 (seeds 1-3, sd 0.0020), a real, seed-verified
  effect, not noise (mean/SE 13.6 against the pre-registered threshold 2.35).
* **H4 is negative.** The same comparison at 15000 steps: `gap15 = -0.0048`
  (`docs/PLAN.md`, both waves of the ladder, jobs 332064 and 333609). By the
  rule fixed before either wave ran (`< +0.010` means gone), the advantage does
  not survive to this horizon, and the full 73242-step pair is not justified.

So "NGD-Pion beats published Pion outright" is not a claim this project's own
data supports, at this batch size. Continuing to chase that specific claim
(more horizon extensions, hoping a longer run reverses the trend) was not
adopted -- the data already answered it, twice, with margin.

Alongside this, the evening's theoretical work (journal, 2026-09-22/23)
produced content that does not fit inside a "does it win" framing at all: the
exact bivector-Fisher derivation and its scale-invariance under `W -> cW`
(proven, not measured); the finding that `alpha = quad/curv` is not a trust
region but a basis-staleness signal, tied to the instantaneous learning rate,
schedule-length-invariant as a fraction of any run; that removing it is
catastrophic (ablation B, 0.356 worse than Pion itself) for a reason not yet
understood (the `t_fac = 1` anomaly: a *fresher*, more accurate basis makes the
loss *worse*); and the noise account (96.4% sampling noise) that explains why
any of this should be batch- and horizon-dependent at all, tying back to
Amari's natural-gradient optimality being an asymptotic, low-noise-regime
result.

## Decision

**The paper is framed around the preconditioner itself** -- what it takes to
build a Fisher/K-FAC-style preconditioner for a spectrum-preserving rotational
optimizer, what it needs to work at all, and under what conditions it helps --
rather than around "NGD-Pion beats Pion." Concretely, the spine becomes:

1. The derivation: the Fisher operator on the bivector tangent space, closed
   form via the K-FAC independence assumption, inversion via a congruence
   basis, and the *exact* (not measured) scale-invariance under `W -> cW` that
   falls out of it for free -- verified algebraically, not empirically.
2. **The trust-region finding, told honestly rather than hidden.** `alpha` is
   necessary (ablation B) but is not what it was originally described as (not
   a trust region; a basis-staleness signal tied to the current learning rate);
   removing it is catastrophic; why it helps at all is not fully understood.
   This is reported as a finding about what this class of preconditioner
   needs, not smoothed over.
3. **The noise/regime account.** The classical natural-gradient advantage is
   asymptotic and low-noise; the step here is 96.4% sampling noise; this
   predicts and explains the batch- and horizon-dependence measured (H1 holds
   at small batch/short horizon, grows with batch, decays with horizon, H4
   negative at length).
4. **A comparison of preconditioners on the same geometry**: Fisher/K-FAC-style
   (this work) against Shampoo-on-so(n) (already run, loses by 0.174 at full
   length but achieves a structurally tighter cross-layer calibration, 2-7x
   against 4658x-36496x) -- what kind of curvature information this class of
   optimizer actually needs.
5. **The positive result, scoped honestly**: a real, seed-verified speedup at
   the classical batch and short-to-medium horizon, reported as a
   step-equivalent effect with its measured boundary (absorbed by 15000
   steps), not as an unconditional claim.

## Consequences

* The full 73242-step pair (`docs/PLAN.md` stage 4) stays not run and not
  planned; nothing in this decision reopens it.
* The remaining open threads (extending `trust="exact"`'s grid below its
  current edge; the pure-K-FAC/`t_fac=1` test discussed the same evening; the
  cross-layer-correction question) are no longer "rescue the headline result"
  work -- they are additional characterization content for the same paper,
  and can be pursued or dropped on their own merits and cost, without pressure
  to produce a win.
* `ALGORITHM.md` and the top-level `README.md`, both still written as if the
  goal were a leaner-or-better-than-Pion optimizer, will need a pass once the
  paper's actual structure is drafted -- not done as part of this decision.

## Evidence

`docs/PLAN.md` (H1, H4, the "What each outcome means for the paper" section,
resolved 2026-09-23); `docs/RESUME.md`'s rewritten "As of 2026-09-23" section;
`docs/JOURNAL.md`, 2026-09-22 (the K-FAC/trust-region theoretical discussion,
the exact scale-invariance derivation, the literature check) and
2026-09-23, 01:20 (H4's resolution). The user's own words, asked directly
whether this framing still satisfies the original ask for a paper with a
result: "да, устраивает, давай так и оформим."
