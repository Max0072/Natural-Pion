# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it and
`docs/JOURNAL_INDEX.md` indexes it, `docs/CLUSTER.md` is the machine,
`$DATA_p330/runs/README.md` says what every run directory holds. Four more
indexes are linked from `AGENTS.md`. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-28, late evening

### The headline: a tie, inside their own noise

Full length, 73242 steps, AdamW pinned at 1e-3:

    pion, eight runs   best 3.3719   median 3.4061   worst 3.4432   spread 0.071
    ngd-pion-s         3.3868 at 96.4%, still falling
    shampoo-pion       3.5456 FINAL
    ngd-pion (S = I)   3.6728 FINAL   (the older run)

**`ngd-pion-s` sits inside `pion`'s own run-to-run spread**, between their
second and third best. Ahead of their median by 0.02, behind their best by
0.01. One run against eight. That is not "we beat Pion" and it is not "we lose
to Pion"; it is a tie, and it should be written as one.

The lead decayed monotonically and was flagged as decaying all day: +0.133 at
step 500, +0.070 at 3000, +0.022 against the median at 70000. **Read the
horizontal gap, not the vertical one** -- in step-equivalent terms it ran
1.2-1.5x for most of the run.

### The distinctive result is not the loss

**`eta` finds itself from a hundredfold range of starting rates.** Adapting it
on the reduction ratio every 5 steps (job 297936):

    eta0     final effective eta    val@3000
    0.002          0.133             4.2771
    0.02           0.152             4.2400
    0.2            0.200             4.3271

A 100x spread collapses to 1.5x and the losses land within 0.087. The claim is
"there is no learning rate to tune", which is a property rather than a
percentage, and it is the most paper-worthy thing the project has.

**It settles on the wrong value -- and the reason is now measured, not
guessed.** With `log_every = 97` (coprime with `t_fac = 25`, see the aliasing
note below), `rho` over 625 measurements separates cleanly:

    eta = 0.02 (good)   0.47 0.33 0.45 0.81 1.02 1.15 1.14 1.35
    eta = 0.15 (bad)    0.04 0.02 0.04 0.06 0.11 0.16 0.25 0.15

So `rho` discriminates, and **K-FAC's band `[0.25, 0.75]` is exactly wrong
here**: at the good rate `rho` sits mostly *above* 0.75, so the rule reads
"be bolder", grows `eta`, and lands in the bad regime. The target should be
`rho ~ 1`: grow above 1.2, shrink below 0.8. **That run has not been made.**

### What is settled, with the number

| | |
|---|---|
| momentum | **+0.045 at 3000 steps**, doubling the margin over `pion` from 0.070 to 0.115. The best arm the project has. **Never run at full length.** |
| the trust region `quad/curv` | load-bearing: removing it costs **0.15** at its own best sampled rate, and `alpha < 1` on 37.1% of layer-steps |
| `quad/curv_exact` per layer | a real trust region, and unusable: the per-layer signal is 8.4x under 109.5x of estimation noise. More tokens cannot close it -- the whole batch buys 5.7x against the 13x needed |
| Shampoo on `so(n)` | **loses by 0.174 at full length**. Delivers the cleanest calibration in the project -- cross-layer angle spread **2.3-2.8x** against the Fisher variant's 4658-36496x, structurally rather than by Pion's `rms` fiat. And it is **prior art**: their own `pion_msign.py` is its memoryless case |
| orthogonal init | 0.117 behind at 3000 steps, consistently |
| no residual connections | nothing trains: AdamW flat at 7.26 for 3000 steps, orthogonal init does not rescue it |
| Levenberg-Marquardt on damping | the rule works (+0.39 over fixed damping) but `damped.py` replaces the spectral floor with it, and that substitution costs 0.72 |
| the step | **96.4% sampling noise** on the real model |
| `kappa` | 1.8e-3 from the `rho` fit, `kfac/exact` 0.0128 from per-token contraction |

### The methodological findings, which may outlast the optimizer

* **A 150-step protocol does not rank these optimizers -- it inverts.** `pion`
  beats `ngd-pion` by 0.30 at 73242 steps and *loses* to it by 0.24 at 150. For
  `shampoo-pion`, `eta = 1e-1` is the worst arm at step 500 and the best from
  step 1000. Every sweep in this repository before 2026-08-27 used 150 steps.
* **Every `rho` ever quoted here was aliased.** `log_every = 100` is a multiple
  of `t_fac = 25`, so every logged row sat on a refactorisation boundary with a
  fresh basis, `alpha = 1` and the lowest `rho` of the cycle. Fixed: the harness
  now reports `rho_med`, `rho_lo`, `rho_hi` over every measurement in the window.
* **Quote the speedup, not the loss difference.** Loss is compressive, so a
  constant step-equivalent advantage shows up as a shrinking gap.
* **Dead-end verdicts in this repository are not evidence.** `damped.py` was
  closed on a mechanism that fired five times in its whole evaluation;
  `ngd-pion-op` was closed on a sweep the journal itself records as unable to
  show an effect. Both were 150-step.

### What to do next, in order

1. **`ngd-pion-m` at full length**, `eta = 2e-2`. It is the best arm at 3000
   steps by 0.045 and it has never been run to length. ~20 rtx-hours. **This is
   the single run most likely to turn the tie into a result.**
2. **The corrected trust-region band**, `rho ~ 1` rather than `[0.25, 0.75]`,
   at 3000 steps first. One hour. If it lands at the swept optimum from a
   hundredfold range of starts, the paper has its distinctive claim.
3. **A tuned AdamW baseline at full length.** There is none, and it is the
   first thing a reviewer asks for. ~10 rtx-hours.
4. **`pion_ablated` at full length.** Still never run, still the isolating
   baseline for any statement about what the preconditioner buys.

### The honest arithmetic on cost

Our step is **2.10x** Pion's (0.967 s against 0.460 on rtx), so a step-count
advantage of 1.2-1.5x is a wall-clock **loss**. The optimizer is 497 ms of a
967 ms step, of which 257 ms is accumulating statistics and 240 ms is the step;
within the step the retraction is 54%, the angle diagnostic 22%, the
factorisation 12%. Newton-Schulz would do the retraction at 0.18x. The
optimizer's cost is per step, not per token, so it amortises with batch: the
ratio falls to 1.29x at 4x the batch and 1.16x at 8x. **None of this has been
done**, and none of it is worth doing until item 1 says whether there is an
advantage to preserve.

### Open bugs

* `ngd_power` is inert for `ngd-pion-s` -- reaches the hash and the directory
  name, not the optimizer.
* `pion_ablated` manifests record the un-ablated momentum, scaling and
  retraction, because the manifest serialises `RunConfig` while
  `build_optimizers` substitutes at construction.
* `basis_congruence` crashed a run at step 41 500; `safe_eigh` has a fallback
  ladder now but the long `ngd-pion-s` run reached 96% without needing it.
* `is_identity` uses `atol = 1e-6` and misses a flat-spectrum fp32 weight at
  1.073e-06.
* `shampoo-pion` does not approach the inert limit monotonically as `eta -> 0`.

### Decided, do not re-litigate

* **The Fisher self-scaling hypothesis is dead**, three measurements.
* **`eta` is a tuned learning rate** -- or an adapted one, see item 2.
* **150 steps cannot rank these optimizers.**
* **Never compare two configurations at a shared `eta`, and never compare a
  grid edge against another arm's optimum.** Both were done today.
* **Concurrent runs on one node must share a seed.**
* **Pin the node for arrays**; partition access and GPU caps here change within
  hours, so re-test rather than trust `docs/CLUSTER.md`.
