# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/PLAN.md` is what to do next and why, `docs/VALIDATION.md` is the
audit of what the numbers are and are not, `docs/JOURNAL.md` is the sequence
that produced all of it and `docs/JOURNAL_INDEX.md` indexes it, `docs/decisions/` holds one record per
decision with its reason and evidence (ADR 0008 is the baseline, 0009 the
comparison protocol), `docs/CLUSTER.md`
is the machine, `$DATA_p330/runs/README.md` says what every run directory
holds. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

Notation, once: **val** is validation loss on held-out C4 (nats per token, lower
is better); **rot** is the learning rate of the rotational optimizer, **adamw**
that of AdamW on the embedding, head and norm gains (56.5% of the model);
**B** is the batch in sequences of 256 tokens, and **B = 512 is the classical
setting** (131 072 tokens per step, 73 242 steps, 9.6B tokens); **gap** is
val(Pion) minus val(NGD-Pion), positive when NGD-Pion is ahead. Every number
below is seed 0, one run per cell, unless it says otherwise.

---

## As of 2026-09-22

Job 332045 is running (stage 1: seeds 1-3 for both arms, B = 512, 3000 steps,
all six concurrent on `rtx6002`, output in `$DATA_p330/runs/seedshead/`). Last
commit before today: 2026-08-31.

### The headline, corrected

The record of 2026-08-29 called the full-length result a tie, "second of nine,
inside Pion's own spread". **That reading was wrong.** Six of the nine were
Pion runs from an earlier harness state (AdamW `beta2` 0.999, norm gains
decayed) that costs about 0.045 by itself, and two were the alternate variant
of Pion. They are not replicates. Details in `docs/VALIDATION.md` V1.

Like-for-like at full length, AdamW pinned at 1e-3 for both:

    ngd-pion-s  rot 0.01   3.3860
    pion, bilateral        3.3719      NGD-Pion behind by 0.014
    pion, alternate        3.3866      level

One run each; the noise at full length is unknown.

### The classical setting, with both optimizers tuned (new since 08-29)

B = 512, 3000 steps (393M tokens, 4.1% of the budget), each arm at its own
optimum of (rot, adamw), both bracketed in adamw:

    ngd-pion-s   rot 6e-3   adamw 8e-3    3.7954
    pion         rot 1e-3   adamw 8e-3    3.8412       gap +0.046

The `adamw2d` sweep is the first with AdamW tuned for either arm; every earlier
run pinned it. It does **not** yet have an error bar: one seed per arm, and no
seed-to-seed spread has been measured at 3000 steps or at full length (only 0.002
of same-seed hardware non-determinism, and sd 0.024 at 150 steps).

Two further readings, each one seed:

* **Horizon:** the gap shrinks with length. With AdamW pinned: +0.024 at 786M
  tokens, -0.014 at 9.6B. At 393M tokens with AdamW tuned: +0.046 (not the same
  protocol, so not a point on the same curve).
* **Batch:** the gap grows with B at fixed tokens: +0.046 at B = 512 against at
  most +0.108 at B = 2048 (tuned; Pion's AdamW optimum at 2048 is still on the
  grid edge, so this is an upper bound), and +0.024 against +0.136 with AdamW
  pinned. **Inside a worse regime:** at the same 393M tokens B = 2048 is 0.26
  worse than B = 512 for NGD-Pion and 0.32 worse for Pion. It is evidence for
  the noise account, not a reason to use the larger batch.

### What the comparison is, exactly

The `pion` arm is **published Pion** (Lie momentum, RMS scaling, truncated
exponential, per-head Q). `ngd-pion-s` differs from it in five things at once,
so the +0.046 is one whole method against another, and no single component is
claimed. **The baseline is published Pion and nothing else** (user, 2026-09-22):
`pion_ablated` is not a baseline, is not planned, and its old 150-step numbers
are void. See `docs/VALIDATION.md` V2 and `docs/PLAN.md` (H3 withdrawn).

### What to do next, in order (full reasoning in `docs/PLAN.md`)

1. **Seeds.** Job 332045, submitted 2026-09-22, three more seeds per arm at the
   two optima above: an error bar on the +0.046. Read the six vals, diff the
   manifests against seed 0, apply the rule in `docs/PLAN.md` H1.
2. ~~`pion_ablated`, tuned.~~ Withdrawn: the baseline is published Pion.
3. **Horizon ladder** at 15000 steps, then the full-length pair with AdamW tuned
   for both, if the trend is not negative.
4. Decide with the user how the paper is framed once 1 and 2 have landed.

### Standing results, with the number

| | |
|---|---|
| momentum | best arm at 3000 steps, **loses at full length**: `ngd-pion-m` 3.3881 / 3.3941 against `ngd-pion-s` 3.3860. Its step-equivalent advantage decayed from 1.60x at 10000 steps to below 1x by the end |
| the trust region `quad/curv` | load-bearing: removing it costs 0.15 at its best sampled rate; `alpha = 1` on only 9.4% of layer-steps at the good rate, median 0.34 |
| the step | **96.4% sampling noise** on the real model (`split_half_step.py`) |
| `beta_D`, `t_fac` cadence | `t_fac` and `beta_D` govern `alpha` as predicted, but removing staleness *costs* loss; `beta_D = 0.9` beats the default 0.5 at every `t_fac` tried: by 0.006 at the current `t_fac = 25`, 0.039 at 1 and 0.080 at 5 (journal, 2026-08-29 evening) |
| true against empirical Fisher | no effect: 3.9091 against 3.9102 at 3000 steps |
| the floor | not eating the Fisher (`iso-cos` 0.82 at the median) |
| Shampoo on `so(n)` | loses by 0.174 at full length; it is prior art (their `pion_msign` is its memoryless case) but the only arm with a cross-layer angle spread near 1 (2.3-2.8x) |
| the adaptive-`eta` controller | converges from a hundredfold range of starting rates to 2-4x above the swept optimum; the losses agree to 0.009. The comparison against a fixed `eta` is void (a diagnostic forward was feeding `A`, fixed 08-29) |
| orthogonal init | 0.117 behind at 3000 steps |
| no residual connections | nothing trains |
| `kappa`, `kfac/exact` | 1.8e-3 and 0.0128: the Fisher does not supply the step scale |

### Methodological findings, which may outlast the optimizer

* **A short protocol does not rank these arms, it inverts them.** `pion` beats
  `ngd-pion` by 0.30 at 73 242 steps and loses by 0.24 at 150. Four orderings
  have now flipped between a short protocol and the full length (journal,
  2026-08-29 evening).
* **Quote the speedup, not the loss difference.** Loss is compressive; a
  constant step-equivalent advantage looks like a shrinking gap.
* **Every `rho` and `alpha` quoted before 2026-08-29 was aliased** (`log_every`
  a multiple of `t_fac`). Use `log_every` 23 or 97.
* **A set of runs is not a set of replicates until the manifests say so.**
  The 08-29 "tie" was built on eight runs from four code states.
* Dead-end verdicts here are not evidence: `damped.py` was closed on a
  mechanism that fired five times, `ngd-pion-op` on a sweep the journal records
  as unable to show an effect. Both were 150-step.

### The honest arithmetic on cost

Our step is **2.10x** Pion's (0.967 s against 0.460 s on rtx), so a step-count
advantage of 1.2-1.5x is a wall-clock loss. The optimizer is 497 ms of the
967 ms step: 257 ms accumulating statistics, 240 ms the step, and within the step
the retraction is 54%, the angle diagnostic 22%, the factorisation 12%. The cost
is per step, not per token, so it amortises with batch: the ratio falls to 1.29x
at 4x the batch and 1.16x at 8x. None of this is worked on, and none is worth it
until the advantage has an error bar.

### Open bugs and traps

* `ngd_power`: commit `40a7886` (2026-08-30) says the exponent now reaches the
  measured-`S` variant. **Not re-tested**; `scripts/sbatch/powercal.sbatch` was
  written and never run.
* `_adamw` uses `cfg.adamw_lr or cfg.lr`, so an unset `--adamw-lr` silently
  means "same as `--lr`". Always pass it.
* `pion_ablated` manifests record the un-ablated momentum, scaling and
  retraction, because the manifest serialises `RunConfig` while
  `build_optimizers` substitutes at construction. A manifest does not tell you
  which arm ran; the code path does.
* `basis_congruence` crashed a run at step 41 500; `safe_eigh` has a fallback
  ladder now.
* `is_identity` uses `atol = 1e-6` and misses a flat-spectrum fp32 weight at
  1.073e-06.
* `shampoo-pion` does not approach the inert limit monotonically as `eta -> 0`.
* the `rho` controller has no dead zone, so the effective rate never settles.
* `ALGORITHM.md` still says `S = I`, no momentum and `t_fac = 100` in its
  decision tables; `AGENTS.md` carries the reversals, the specification does not.
* Five `pion` B = 512 cells in `runs/adamw2d` are empty logs stopped at step 0
  (node hangs). They are not results.
* Nodes: `rtx6002` is the one that has been reliable; `rtx6001`, `rtx6003` and
  `rtx6004` have wedged or packed jobs onto shared GPUs. Re-test before relying
  on any of this.

### Decided, do not re-litigate

* **The Fisher self-scaling hypothesis is dead**, three measurements.
* **`eta` is a tuned learning rate**; adapting it on `rho` converges, but to the
  wrong place.
* **150 steps cannot rank these optimizers**; nor, it now seems, can 3000.
* **Never compare two configurations at a shared `rot`, and never compare a grid
  edge against another arm's optimum.**
* **Concurrent runs on one node must share a seed.**
* **Pin the node for arrays**; partition access and GPU caps here change within
  hours.
* **The anchor was accepted by override on 2026-08-26**: level reproduced to
  0.7-0.9%, arm gap 1.9x theirs. It licenses comparisons inside this harness and
  nothing else; the paper states it as a limitation.
