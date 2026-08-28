# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it and
`docs/JOURNAL_INDEX.md` indexes it, `docs/CLUSTER.md` is the machine,
`$DATA_p330/runs/README.md` says what every run directory holds. Four more
indexes are linked from `AGENTS.md`. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-29, after midnight

### The headline: a tie, and the number is final

Full length, 73242 steps, AdamW pinned at 1e-3. Nine runs exist; sorted:

    3.3719  pion
    3.3860  <- ngd-pion-s, eta = 0.01      FINAL
    3.3866  3.3937  3.4059  3.4062  3.4080  3.4414  3.4432   pion

    ngd-pion-s eta = 0.03   3.4343  FINAL
    shampoo-pion            3.5456  FINAL
    ngd-pion (S = I)        3.6728  FINAL   (the older run)

**`ngd-pion-s` sits inside `pion`'s own run-to-run spread**, second of nine.
Ahead of their median by 0.020, behind their best by 0.014. One run against
eight. That is not "we beat Pion" and it is not "we lose to Pion"; it is a tie,
and it should be written as one. The lead decayed monotonically all the way:
+0.133 at step 500, +0.070 at 3000, +0.020 at the end. **Read the horizontal
gap, not the vertical one** -- in step-equivalent terms it ran 1.2-1.5x for
most of the run.

### Momentum is the live hope, and it is ahead at every matched step

`ngd-pion-m` at full length is 9.3% in (job 298784, rtx6003, ~18h left):

    step      ngd-s   ngd-m 1e-2   diff
     500     4.6269     4.3435    -0.283
    2000     4.0215     3.9404    -0.081
    4000     3.8709     3.8139    -0.057
    6500     3.7995     3.7414    -0.058

The gap compresses to 0.058 and then holds flat for 3000 steps, which is the
signature of a constant step-equivalent advantage rather than a fading head
start. `eta = 1e-2` leads `2e-2` by 0.024 -- the reverse of the 3000-step
sweep, and the third time a short protocol has reordered arms.

### The adaptive-`eta` claim, now measured rather than hoped

Target `rho ~ 1` (grow above 1.2, shrink below 0.8), 3000 steps, 100x range of
starts:

    eta0     final effective eta    val@3000
    0.002          0.089             4.1196
    0.02           0.030             4.1238
    0.2            0.077             4.1150
    control, fixed 0.02              3.9184
    old band [0.25, 0.75], best      4.2400

The controller reaches its target -- `rho_med` is 1.00 from step 300 in all
three arms -- and the losses now agree to **0.009** where they spread 0.087
under the old band. But it lands 2-4x above the swept optimum and pays **0.20**
for it, and the band is not the reason:

* final rates differ 3x across arms while losses differ 0.009, so the outcome
  does not depend on where it lands;
* the arm closest to the swept optimum was the worst of the three.

**`rho` saturates.** It reads 0.02-0.25 at a catastrophic rate and 1.0 at both
ends of the interval that costs 0.20. It asks whether the quadratic model
predicted the drop, not whether the drop was the largest available. **No
controller on `rho` alone will find the swept rate**, and no choice of band
changes that. (Separate defect, worth fixing anyway: 1.5x every 5 steps with no
dead zone, so the rate dithers 0.051/0.077/0.115 forever instead of settling.)

So the paper's distinctive claim is weaker and more precise than "there is no
learning rate to tune": **the rule converges reliably from a hundredfold range
to a rate 2-4x too large, at a cost of 0.20 at 3000 steps.** It removes the
search and pays for it. Strengthening it needs a signal that does not saturate
-- a directional derivative along the step, or achieved-over-predicted measured
at two step lengths. That is a design question, not a run to launch.

### What is settled, with the number

| | |
|---|---|
| momentum | best arm in the project: -0.058 against `ngd-pion-s` at matched steps and holding, at full length, in flight |
| the trust region `quad/curv` | load-bearing: removing it costs **0.15** at its own best sampled rate, and `alpha < 1` on 37.1% of layer-steps |
| `quad/curv_exact` per layer | a real trust region, and unusable: the per-layer signal is 8.4x under 109.5x of estimation noise. More tokens cannot close it -- the whole batch buys 5.7x against the 13x needed |
| Shampoo on `so(n)` | **loses by 0.174 at full length**. Delivers the cleanest calibration in the project -- cross-layer angle spread **2.3-2.8x** against the Fisher variant's 4658-36496x, structurally rather than by Pion's `rms` fiat. And it is **prior art**: their own `pion_msign.py` is its memoryless case |
| orthogonal init | 0.117 behind at 3000 steps, consistently |
| no residual connections | nothing trains: AdamW flat at 7.26 for 3000 steps, orthogonal init does not rescue it |
| Levenberg-Marquardt on damping | the rule works (+0.39 over fixed damping) but `damped.py` replaces the spectral floor with it, and that substitution costs 0.72 |
| the step | **96.4% sampling noise** on the real model |
| `kappa` | 1.8e-3 from the `rho` fit, `kfac/exact` 0.0128 from per-token contraction |

### The methodological findings, which may outlast the optimizer

* **A short protocol does not rank these arms -- it inverts.** `pion` beats
  `ngd-pion` by 0.30 at 73242 steps and *loses* by 0.24 at 150. For
  `shampoo-pion`, `eta = 1e-1` is the worst arm at step 500 and the best from
  1000. For `ngd-pion-m`, `2e-2` wins at 3000 and loses from 500 onward at
  length. Every sweep here before 2026-08-27 used 150 steps.
* **Every `rho` quoted before 2026-08-28 was aliased.** `log_every = 100` is a
  multiple of `t_fac = 25`, so every logged row sat on a refactorisation
  boundary with a fresh basis, `alpha = 1` and the lowest `rho` of the cycle.
  Fixed: the harness reports `rho_med`, `rho_lo`, `rho_hi` over the window.
* **Quote the speedup, not the loss difference.** Loss is compressive, so a
  constant step-equivalent advantage shows up as a shrinking gap.
* **Dead-end verdicts in this repository are not evidence.** `damped.py` was
  closed on a mechanism that fired five times in its whole evaluation;
  `ngd-pion-op` was closed on a sweep the journal records as unable to show an
  effect. Both were 150-step.

### What to do next, in order

1. **Let `ngd-pion-m` finish** (~18h). It is the only thing in flight and the
   single result most likely to turn the tie into a win. Nothing else should
   compete with it for rtx6003.
2. **A tuned AdamW baseline at full length.** There is none, and it is the
   first thing a reviewer asks for. ~10 rtx-hours.
3. **`pion_ablated` at full length.** Still never run, still the isolating
   baseline for any statement about what the preconditioner buys.
4. **A non-saturating step-size signal**, if the adaptive claim is to be
   strengthened. Design first, measure on the 3000-step protocol second.

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
  ladder now and the full-length `ngd-pion-s` run never needed it.
* `is_identity` uses `atol = 1e-6` and misses a flat-spectrum fp32 weight at
  1.073e-06.
* `shampoo-pion` does not approach the inert limit monotonically as `eta -> 0`.
* the `rho` controller has no dead zone, so the effective rate never settles.

### Decided, do not re-litigate

* **The Fisher self-scaling hypothesis is dead**, three measurements.
* **`eta` is a tuned learning rate**; adapting it on `rho` converges but to the
  wrong place, and `rho` cannot be made to find the right one.
* **150 steps cannot rank these optimizers.**
* **Never compare two configurations at a shared `eta`, and never compare a
  grid edge against another arm's optimum.**
* **Concurrent runs on one node must share a seed.**
* **Pin the node for arrays**; partition access and GPU caps here change within
  hours, so re-test rather than trust `docs/CLUSTER.md`.
