# ADR 0013 -- the Pion baseline runs the published recipe, with no extra knobs (no warmup)

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user

## Context

The trust-region audit found that `alpha = quad/curv` behaves like an implicit,
statistics-driven warmup for NGD-Pion (median 0.001 in the first 100 steps, 1.000
in the last 500), while both arms run with `warmup_steps = 0` as in Pion's own
script. The gap to Pion is made early (+0.179 at step 500, +0.038 at 3000, mean of
four seeds). That suggested testing Pion with a tuned warmup, so that a method
with an implicit warmup is not compared with a baseline that has none.

## Decision

**Not done.** The baseline is Pion with the published recipe and nothing added.
Warmup in particular: the user's reason is that there is nothing in Pion to warm
up. Its rotational step is normalised from the first step (the RMS of the update
per element is fixed at `lr * 0.2` whatever the gradient) and the transformation is
orthogonal, so neither the spectrum nor the norm of a weight can blow up. The one
place a warmup is classical is the AdamW half of the model (embedding, head, norm
gains, 56.5% of the parameters), and the authors' script has none there either.

The arms keep being tuned on the two rates already in ADR 0009, `rot` and `adamw`,
and on no third.

## Consequences

The mechanism question about `alpha` is asked on the NGD-Pion side only: does an
explicit warmup reproduce what `alpha` does when `alpha` is switched off (ablation
B and its follow-up, `docs/PLAN.md` H5). Pion is not touched. If a reviewer asks for
Pion with a warmup, it is a 25-minute run and is made then. The earlier proposal to
give Pion a tuned warmup, made in this session as a fairness measure, was a guess
about a gap and not evidence that Pion loses anything early; no run of published
Pion shows that.

## Evidence

`ngd_pion/pion_baseline.py` and the Pion paper (RMS scaling, exact generators);
`harness/config.py` (`warmup_steps = 0`, the setting of their 60M script);
journal 2026-09-22, the two entries on `alpha`.
