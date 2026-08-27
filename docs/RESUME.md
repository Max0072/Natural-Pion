# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it (newest last, and long
by now), `docs/CLUSTER.md` is the machine, `$DATA_p330/runs/README.md` says
what every run directory holds. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-27, evening

### In flight

Nothing. Jobs 273026 (the `eta` sweep) and 273528 (the K-FAC probe) both
finished; both are written up in the journal entry of 2026-08-27.

### The headline number, and it is against us

Full length, 73 242 steps, AdamW pinned at 1e-3 in both arms:

| run | val |
|---|---|
| `ngd-pion` eta 1.0, T_fac 25 | **3.6728** |
| `pion` (their published config) | 3.3719, 3.3866 |

**We lose by 0.30.** The published gap between Pion's own two arms is 0.0079,
and repeat runs of an identical configuration differ by about 0.002, so this is
far outside noise.

**The comparison isolates nothing.** `pion` carries `momentum="lie"`,
`scaling="rms"` and a truncated retraction; NGD-Pion has none of the three. The
isolating baseline is `pion_ablated` and **no full-length run of it exists** --
that is still the single most important missing measurement. At 150 steps
NGD-Pion beat `pion_ablated` by 6 sd; at full length there is nothing to
compare with.

### The measured `S` wins, once given its own `eta`

Job 273026: `ngd-pion-s`, MC off, `power = 1`, `T_fac = 25`, AdamW pinned at
1e-3, seed 0, 150 steps. Optimum bracketed on both sides.

    eta      val@150
    3e-4      5.6987
    1e-3      5.6290
    3e-3      5.5702
    1e-2      5.5068   <- minimum
    3e-2      5.6249
    1e-1      5.8845

Against `ngd-pion` (`S = I`) at its own swept optimum over 0.003 to 300:
**5.74 - 5.83**. One variable, and the measured `S` wins by **0.23 to 0.32**
against a 0.07 noise floor. It also beats the MC-sampled arm at the same `eta`
(5.5517), so MC sampling contributes nothing and the win belongs to `S`.

**The old claim that `S = I` is better is refuted.** The measured `S` had never
been given its own `eta`, which is a hundred times smaller -- exactly what
`with_s.py` predicted in its docstring.

### `alpha` is not vestigial, and `eta* = 2` is fully accounted for

The old "decided" entry claiming `alpha == 1.000` for every `T_fac <= 10` does
not hold at `T_fac = 25`: it runs 2.6e-4 to 0.19, cutting the step by three to
four orders every step.

`rho` is clean of AdamW -- `train.py` measures `after` between `rot.step()` and
`adamw.step()`, deliberately. What biases it is `clip_grad_norm_`, which runs
before `rot.step()`: the clip scales `G` by `s`, `X` is linear in `G`, so
`quad` goes as `s^2` while the actual decrease goes as `s`, leaving a constant
factor `1/s` on `rho`, identical across arms at step 0.

    eta      rho@0    angle_max@0   val@150
    3e-4     2.1677      1.1 rad     5.6987
    1e-3     1.9831      3.6         5.6290
    3e-3     1.3075     10.8         5.5702
    1e-2     0.4728    ~36           5.5068   <- best loss
    3e-2     0.1511   ~108           5.6249

`rho = 2.2822 - 322.8 lr` fits the first three arms to +-0.02 and misses the
next two by +1.42 and +7.55, so the second-order model holds only to about
`eta = 3e-3`; past it `rho` saturates rather than going negative.

**The best loss sits where the model over-predicts twofold.** `ALGORITHM.md`
justifies dropping Pion's RMS scaling as "a testable hypothesis: the Fisher
sets the scale itself". That test has now been run and the hypothesis fails.
`eta` is a tuned learning rate; say so in the paper rather than working around
it.

**Where the 500x went.** With `kappa := curv/(4 Q)` the quadratic optimum is
`c* = 2 kappa`, so the theoretical `eta* = 2` assumes `kappa = 1`. Measured
`kappa = 1.8e-3` gives `c* = 3.5e-3` against an observed 1e-2 -- the whole gap,
to within a factor of 3 over five orders.

**And it is the independence assumption.** Job 273528 ran
`scripts/probes/kfac_error.py` for the first time, on fresh weights:
`kfac/exact` = 3.11e-2, 6.04e-3, 1.13e-2, geometric mean **0.0128**. Two orders
below 1, so the gap is not Fisher-against-Hessian. The in/out cross term is
excluded separately by Cauchy-Schwarz (`kappa >= 1/2` whatever the data). The
pre-registered bound of 0.0036 was missed by 3.6x, so independence is most of
`kappa`, not all of it.

### What to do next, in order

1. **Exact `curv`, so `alpha` means something.** Build the direction with the
   approximation, measure the length with the truth, as K-FAC does:
   `curv_exact = E_b[(2 u_b^T X x_b)^2]`. One `(tokens x n)` matmul and a
   contraction per layer per step, no extra pass -- `u_b` and `x_b` are already
   in the hooks. Prediction, stated before writing it: `alpha` should land near
   1e-2 on a fresh basis and `rho` at `eta = 1` should come out near the 0.47
   that `eta = 1e-2` shows now. If `alpha` lands near 1, the implementation is
   wrong, not the theory.
2. **`pion_ablated` at full length.** About 10 rtx-hours. Without it nothing at
   full length is interpretable, and it has been the top blocker for days.
3. **`ngd-pion-s` at full length at `eta = 1e-2`**, now that the optimum is
   bracketed. This is the arm the paper would actually report.
4. **Momentum.** NGD-Pion has none; Pion has it in the Lie algebra. Not a
   confound to argue away, a missing part of the method.

### Open bugs

* **`ngd_power` is silently inert for `ngd-pion-s`.** It reaches the manifest,
  the run hash and the directory name without reaching the optimizer, so it
  reads as a controlled variable and is not. Either wire it or reject it in
  `RunConfig`.
* `basis_congruence` crashed a run at step 41 500 -- `linalg.eigh` failing to
  converge on an ill-conditioned pencil. It will bite **more often** under
  `ngd-pion-s`, where the out-side takes the congruence path too.
* `is_identity` uses `atol = 1e-6` and a flat-spectrum 512x512 weight built in
  fp32 misses it at `1.073e-06`. Nothing takes the cheap basis path that
  should. It should also accept a matrix proportional to the identity, which is
  what a flat initialisation gives a non-square layer.
* **The out-side solves in a space the problem does not occupy.** For `m > n`
  the null space of `F_out` is exactly `so(range(W)^perp)`, dimension
  `(m-n)(m-n-1)/2` -- 39% of `so(1376)` on the FFN layers, and `eps` is applied
  there for nothing. Note the journal's earlier phrasing, "the meaningful
  problem is `so(n)`", is **wrong**: the `(kernel, range)` block is live and
  carries ~88% of the step. The correct target is `so(m)/so(m-n)`, the Stiefel
  tangent space, of dimension `mn - n(n+1)/2`. Hygiene and cost, not a fix:
  in exact arithmetic with a fresh basis the step is unchanged, because the
  `(kernel, kernel)` numerator is exactly zero.

### Decided, do not re-litigate

* **The anchor is accepted** against its pre-registered criteria, deliberately,
  by the user. Level reproduces to 0.7%; the arm gap comes out 1.9x theirs. It
  licenses internal comparisons in this harness and not the quoting of our
  numbers as reproductions of theirs.
* **Pin `--adamw-lr` for any sweep over `lr`.** One rate drove both optimizers
  until 2026-08-26, so every learning-rate conclusion before that date measured
  AdamW. `adamw_lr = 0` still means "follow lr", which is their published
  design, so the anchor is unaffected.
* **Never compare two optimizers at a shared `eta`.** Each has its own optimum
  -- on the toy problem they spanned 4.6e-3 to 2.2e+1 -- and this project has
  now walked into it four separate times: the initialisation sweep, the
  heavy-tail sweep, the `ngd-pion-op` null result, and the standing claim about
  the measured `S`.
* **Concurrent runs on one node must share a seed.** Different seeds read
  different corpus windows and halve each other's IOPS: 18.7 s/step for two,
  4.4 as soon as one is cancelled. Sweep seeds sequentially.
* **A single 150-step run resolves about 0.05 in loss.** sd is 0.024 over four
  seeds; repeats of `ngd-pion` at `eta = 1` span 5.7621 to 5.8308, so treat
  0.07 rather than 0.024 as the practical floor.
* **`T_fac` around 25.** 100 is clearly worse; below 25 nothing is resolvable
  and the cost keeps climbing.
* **The covariance bf16 bug is fixed** (`59109d3`) and **every run now on disk
  postdates it**. `n_negative` is 0 across all 56 weights in the recent runs.
  Do not re-diagnose negative curvature from that cause.
