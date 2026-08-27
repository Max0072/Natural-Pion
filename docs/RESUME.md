# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it (newest last, and long
by now), `docs/CLUSTER.md` is the machine, `$DATA_p330/runs/README.md` says
what every run directory holds. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-27, evening

### In flight

`eta` sweep for `ngd-pion-s` with MC sampling off -- see "What to do next".

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

### The best 150-step number we have is an `S` variant

| run | eta | val@150 |
|---|---|---|
| `ngd-pion-s` + MC sampling (`mcfisher2`) | **0.01** | **5.5517** |
| `ngd-pion` (dense sweep, 0.003 to 300) | 1 - 30 | 5.74 - 5.83 |
| `ngd-pion-pow` | 0.01 | 5.8240 |
| `ngd-pion-s`, no MC | 1 | 6.1019 |

Read this carefully, because the obvious reading is wrong twice.

* **`eta = 0.01` is the bottom of that grid** (`{0.01, 0.5, 2, 8}`). The
  optimum is not bracketed.
* **`mcfisher2` differs from every other arm in more than one way.** Its
  manifest records `ngd_power: 0.5`, which **had no effect** -- `power` is
  passed only for `ngd-pion-pow`, so `FastNGDPionS` ran at power 1. What it
  does carry that the others do not is MC sampling of the labels. So 5.5517 is
  "`S` measured, power 1, MC on" and the 0.28 it wins by is not attributed.

### What is actually established about `S`

This file used to say "`S = I` is why we lose". That is not supported.

`with_s.py` states in its own docstring that `eta` does not carry over and must
be swept from scratch. **It never was.** Every non-MC `ngd-pion-s` run sits at
`eta = 1`, and the sweep launched for it (`with-s`, `eta` 1e-8 to 1e-14) has
`steps = 0` and `val = nan` in all four directories -- it died before logging a
step. The claim that the measured `S` is worse rests on a single point at the
one `eta` the module says will be wrong.

What *is* measured: the per-layer allocation **inverts** rather than narrows.
`delta_rms` spans ~1200x across the 56 weights in every run regardless of
optimizer, and

    family                        rho(log angle, log delta_rms)
    S = I    (ngd-pion/-op/-pow)          +0.81 .. +0.92
    S measured (ngd-pion-s)               -0.37 .. -0.65

so under the measured `S` the biggest steps go to the layers whose backward
signal -- and therefore whose `D` estimate -- is weakest. Whether that is
harmful is exactly what is not yet known.

### `alpha` is NOT vestigial, and `rho = 0.083` on a fresh basis

The old "decided" entry claiming `alpha == 1.000` for every `T_fac <= 10` does
not hold for `ngd-pion-s` at `T_fac = 25`:

    step   alpha_min   alpha_max      rho    angle_max
       0      1.0000      1.0000    0.083         67
      90      2.6e-04      0.088    0.282        6.5
     149      7.2e-04      0.185    0.204        5.1

The trust region cuts the step by three to four orders every step, and the
quadratic model still over-predicts the decrease fivefold. `rho` is an
over-estimate of the rotational part: the numerator is the whole model's
decrease, AdamW included, the denominator only the rotational prediction.

**Step 0 is the number to explain.** Fresh basis, `alpha = 1.000`, where
`quad = curv` is an identity -- and the loss falls by 8% of what the descent
lemma predicts, with `angle_max = 67` radians. Nothing about staleness,
degeneracy, damping or the choice of `S` is involved there. A scalar `alpha`
cannot fix it: it shortens the step uniformly while the problem is that the
*direction* is dominated by the flattest directions of `F`.

### What to do next, in order

1. **`ngd-pion-s`, MC off, swept over `eta`** around 0.01. One variable against
   `ngd-pion`'s existing `eta` curve, and it separates `S` from MC sampling.
   Half an hour of rtx. This is cheap and it is blocking the interpretation of
   every `S` result.
2. **`pion_ablated` at full length.** About 10 rtx-hours. Without it nothing at
   full length is interpretable. Its own `eta` has never been found beyond 150
   steps, where it was 0.5.
3. **Bracket `eta*` for the MC arm** below 0.01, which the `mcfisher2` grid did
   not do.
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
