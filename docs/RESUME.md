# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it (newest last, and long
by now), `docs/CLUSTER.md` is the machine, `$DATA_p330/runs/README.md` says
what every run directory holds. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-27, night

### In flight

Nothing on the cluster. Jobs 276154, 278352 (the five-arm array) and 278521
(the validation re-run) all finished.

### The direction changed today

The Fisher preconditioner is no longer the thing being repaired.
`ALGORITHM.md` proposed as a testable hypothesis that the Fisher sets the step
scale itself -- hence no RMS scaling, and `eta* = 2` derived rather than tuned.
Three independent measurements now refute it: `kappa = 1.8e-3` from the `rho`
fit, `kfac/exact = 0.0128` from `kfac_error.py`, and `alpha_exact/alpha` of
5e-3 to 1e-1 from job 274008. **`eta` is a tuned learning rate; the paper has
to say so.** The warmup rescue was tested at its predicted optimum (job 274907,
`eta = 0.15`) and lost by 0.38.

The decision, taken by the user, is to keep Pion's rotational geometry and
replace the preconditioner with **Shampoo's**, built from the generators
themselves rather than from network statistics. Implemented today, and job
276154 is its first GPU reading. The pre-registration was written before any
compute; see the journal entries of 2026-08-27 evening and night.

**Prediction 1 landed.** Cross-layer spread of rotation angle, sustained from
step 30 to 150: **2x to 7x**, against the Fisher variant's 4658x-36496x. The
defect the full-length run diagnosed is fixed structurally, where Pion buys the
same calibration by fiat with `scaling="rms"`. `eta` also has a direct physical
reading now -- it *is* the first step's rotation angle in radians.

**`eta` is now swept and bracketed on both sides**, minimum at `1e-2`:

    3e-4 6.1414   1e-3 6.1048   3e-3 6.1589   1e-2 5.9795   3e-2 6.1690   1e-1 6.8903

Prediction 1 holds across the whole bracket (spread 2.3x-2.8x), degrading to
12.9x only at `1e-1` where the method breaks. So `shampoo-pion` at its own
optimum is **5.9795**, against `pion_ablated` 6.1143, `ngd-pion` 5.9113 and
`ngd-pion-s` 5.5068, each at its own: it beats the isolating baseline by 0.13,
ties `ngd-pion`, and trails `ngd-pion-s` by 0.47. All 150 steps; none of them
is a full-length result.

**The optimum was re-run because it was the odd arm out** -- different node,
different day, plane diagnostic on. Job 278521 reproduced it to four decimals,
which settles that the dip is real, that `_diagnose` is read-only as designed,
and that this optimizer is run-to-run deterministic at 150 steps. Do not assume
the 0.07 practical floor applies here; it was measured from `ngd-pion` repeats.

**New lever, promoted by the data.** 62-78% of the accumulator's spectrum sits
on the relative floor once it mixes directions (partly by construction:
`G_out` for a (1376,512) weight has rank at most 1024 inside `so(1376)`). In
the Fisher path every `eps` sweep came back null; here `shampoo_eps` governs
the majority of the spectrum and is the second thing to sweep.

**Step cost is measured and acceptable: 1.14 s/step** with five arms sharing a
node, against a compute-bound floor near 1 s and `ngd-pion-s`'s 0.98 s warm and
solo. At most 16% slower, and cheaper in memory (82.5 GB against 83.3). The
five-arm sweep took 5:55 against 26:24 for the sequential six-arm Fisher sweep.
The plane diagnostic alone costs 4.6x the step, which is why it is opt-in.
Warm-up is the corpus, not the code -- `TokenCorpus` draws 512 random windows a
step from a 20 GB file, so concurrent arms must share a seed to warm each
other's page cache rather than compete.

### What is new in the code

* `ngd_pion/shampoo.py` -- `ShampooPion`, registered as `shampoo-pion`.
  No hooks, no covariance, no backward recorder: it is a pure function of the
  gradients it has seen. `power = 0` is exactly ablated Pion and is the control.
* `ngd_pion/shampoo_reference.py` -- the numpy oracle, in the role
  `reference.py` plays for NGD-Pion.
* `tests/test_shampoo.py` -- 17 tests. Suite is now **204 passed, 1 skipped**.
* `harness/config.py` -- `shampoo_power`, `shampoo_beta`, `shampoo_damping`,
  `shampoo_eps`, `shampoo_plane_every`.
* `harness/train.py` -- `DIAGNOSED` replaces two `isinstance(rot, NGDPion)`
  gates that had silently written no diagnostics at all for the new optimizer.

### Why so(n) suits Shampoo, in one paragraph

For skew `G`, `G G^T = -G^2 = G^T G`, so Shampoo's two factors coincide: one
accumulator and one `eigh` instead of two. The two-sided sandwich
`P^-1/4 G P^-1/4` is skew, so the step stays a rotation generator and Cayley
still freezes the spectrum -- the one-sided form does not, except on the first
step where `P` commutes with `G`, which is a trap an implementation checked
only at step 0 would fall into. With a single accumulated gradient the sandwich
orthogonalises the generator exactly, so every rotation plane turns by the same
angle. All three are pinned to machine precision in the tests.

### What to do next, in order

1. **`shampoo_eps`, on a two-dimensional grid.** The relative floor is not just
   damping here: at `eps = 1` the spectrum collapses to `lam_max`, `Q` becomes
   proportional to the identity and the step is the raw generator -- ablated
   Pion up to a scalar -- while `eps -> 0` is full Shampoo. So `eps` traces a
   continuous path from the control to the method. **It cannot be swept at
   fixed `eta`**: `||X||` depends on `eps`, so a 1-D sweep measures a rescaling
   that `eta` re-absorbs, which is exactly why the `ngd-pion-op` damping sweep
   came back null and was misread. Grid `eps x eta`, five arms per node at
   ~6 minutes a batch.
2. **`pion_ablated` at full length**, about 10 rtx-hours. **Unaffected by the
   pivot and now more necessary, not less**: whatever drives the rotation, that
   is the isolating baseline, and no full-length number means anything without
   it. It has been the top blocker for days.
3. `shampoo-pion` at full length, if and only if step 1 brackets an optimum
   that beats `pion_ablated` at its own.

### Pre-registered for `shampoo-pion`, before any GPU time

* ~~Cross-layer spread of `angle` falls from ~1e4 to order 1e1 or below.~~
  **Met**: 2x-7x, job 276154.
* `eta` shows a broad plateau rather than the sharp optimum at 1e-2.
  **Not confirmed.** The curve spans 0.19 over `3e-4` to `3e-2`, the same range
  `ngd-pion-s` spans -- but jagged rather than U-shaped, with the four
  non-optimal arms bunched in 0.064 and 1e-3 beating 3e-3.
* **Named risk:** raw Shampoo without grafting is known to be unreliable in
  step scale. If the arm blows up, the first hypothesis is scale, and grafting
  the norm -- the analogue of Pion's `rms` -- is the named fallback rather than
  a rescue invented afterwards.
* ~~A toy CPU run gave a cross-layer spread of 1.6x, which is a wiring check.~~
  Superseded: the real model gives 2x-7x, which is the reading that counts.

### Open bugs

* **`ngd_power` is silently inert for `ngd-pion-s`.** It reaches the manifest,
  the run hash and the directory name without reaching the optimizer, so it
  reads as a controlled variable and is not. Either wire it or reject it in
  `RunConfig`. `test_shampoo.py` now pins the equivalent wiring for the new
  optimizer so the class of bug does not recur there.
* `powered.py`'s docstring calls `power = 1/2` "Adam, Shampoo". The exponents
  agree arithmetically; the matrices being raised to them do not. Correct it.
* `basis_congruence` crashed a run at step 41 500 -- `linalg.eigh` failing to
  converge on an ill-conditioned pencil. `safe_eigh` now has a three-rung
  fallback, but **the long `ngd-pion-s` run has still never been attempted**
  and this is where it would bite. Shampoo does not use that path at all.
* `is_identity` uses `atol = 1e-6` and a flat-spectrum 512x512 weight built in
  fp32 misses it at `1.073e-06`.
* **The out-side solves in a space the problem does not occupy.** For `m > n`
  the null space of `F_out` is `so(range(W)^perp)` -- 39% of `so(1376)` on the
  FFN layers. Hygiene and cost, not a fix. Applies to the Fisher path only.

### Decided, do not re-litigate

* **The anchor is accepted** against its pre-registered criteria, deliberately,
  by the user. It licenses internal comparisons in this harness and not the
  quoting of our numbers as reproductions of theirs.
* **The Fisher's self-scaling hypothesis is dead**, by three independent
  measurements. Do not re-derive `eta* = 2`.
* **The toy is not the model.** A two-layer hidden-64 run produced the
  attention/FFN 1000x split (did not reproduce, came out 2-5x) and now the 1.6x
  angle spread (unverified). Toy readings are wiring checks.
* **Pin `--adamw-lr` for any sweep over `lr`.** One rate drove both optimizers
  until 2026-08-26, so every learning-rate conclusion before that date measured
  AdamW.
* **Never compare two optimizers at a shared `eta`.** Each has its own optimum;
  this project has walked into it four separate times.
* **Concurrent runs on one node must share a seed.** Different seeds read
  different corpus windows and halve each other's IOPS.
* **A single 150-step run resolves about 0.05 in loss**; treat 0.07 as the
  practical floor.
* **`T_fac` around 25.**
* **The covariance bf16 bug is fixed** (`59109d3`) and every run now on disk
  postdates it.

### The headline numbers, for reference

Full length, 73 242 steps, AdamW pinned at 1e-3:

| run | val |
|---|---|
| `ngd-pion` `S = I`, eta 1.0, T_fac 25 | **3.6728** |
| `pion` (their published config) | 3.3719, 3.3866 |

**The comparison isolates nothing**, and is wrong in three ways at once: the
worse variant (the measured `S` beats `S = I` by 0.23-0.32), off its own
150-step optimum of `eta = 3`, and against a baseline carrying momentum, RMS
scaling and a truncated retraction. At 150 steps, best against best,
`ngd-pion` beat `pion_ablated` 5.9113 to 6.1143.
