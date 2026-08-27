# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it (newest last, and long
by now), `docs/CLUSTER.md` is the machine, `$DATA_p330/runs/README.md` says
what every run directory holds. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-28, small hours

### In flight

Nothing. The crossover study (jobs 279222, 279617) finished.

### The headline, and for once it is in our favour

**`ngd-pion-s` is ahead of `pion` at every horizon that has been measured.**
Against the eight full-length `pion` runs already on disk -- configuration
checked field by field, not assumed: same `adamw_lr` 1e-3, `adam_beta2` 0.95,
`decay_norms_and_biases` False, seed, precision --

    step               500     1000     1500     2000     2500     3000
    pion best of 8   4.7275   4.3150   4.1671   4.0781   4.0272   3.9880
    pion median      4.8710   4.4212   4.2173   4.1241   4.0571   4.0133
    ngd-pion-s       4.5947   4.2177   4.0866   4.0124   3.9619   3.9184
    lead over best  +0.1328  +0.0973  +0.0805  +0.0657  +0.0653  +0.0696

The lead narrows to about 0.066 and then stops narrowing, holding 0.065-0.070
from step 2000 to 3000; against the median of the eight it is 0.095 at 3000.
This is the first time anything in this project has beaten `pion` like for
like. It is **one run against eight, at 4% of the schedule**, and it is not a
full-length result.

### The methodological finding, which is the bigger one

**Every ranking this repository has taken at 150 steps is unreliable, and the
short-horizon choice can be exactly inverted.**

Two independent demonstrations:

* `pion` and `ngd-pion` are the only optimizers measured at both horizons. At
  150 steps `ngd-pion` wins by 0.24; at 73 242 steps `pion` wins by 0.30. The
  ordering reverses.
* For `shampoo-pion` at `eps = 1e-6`, `eta = 1e-1` is the **worst** arm at step
  500 and the **best** from step 1000 onward, while the 150-step optimum of the
  same row was `3e-3`. `eta*` moves thirtyfold with horizon.

So: **stop choosing `eta` at 150 steps.** Every hyperparameter this project has
selected -- the initialisation sweep, the heavy-tail sweep, `S = I` against the
measured `S`, `T_fac`, the `eta` and `eps x eta` grids -- was selected with an
instrument that inverts on the one case where it can be checked. 3000 steps
costs about 50 minutes on one GPU and five arms fit in the 8-GPU quota.

### Where Shampoo stands

`shampoo-pion`, `eps = 1e-6`, is behind `pion` but closing:

    eta        500     1000     1500     2000     2500     3000
    1e-2     5.1617   4.8495   4.7202   4.6456   4.5943   4.5593
    3e-2     5.0759   4.7265   4.5858   4.5019   4.4434   4.4006
    1e-1     5.3561   4.6801   4.4733   4.3586   4.2485   4.1778

Its deficit against `pion`'s best goes 0.63, 0.37, 0.31, 0.28, 0.22, 0.19 --
steadily closing, and `1e-1` is still the largest `eta` sampled, so every
number is an upper bound. It is not competitive with `ngd-pion-s` on anything
measured. **Do not extrapolate either trend**; that is precisely the error the
150-step protocol turned out to embody.

What Shampoo does deliver, and it is a real result: the cross-layer spread of
rotation angle falls from the Fisher variant's 4658x-36496x to **2.3x-2.8x**,
structurally, where Pion buys the same calibration by fiat with
`scaling="rms"`. Step cost 1.14 s against 0.98 for `ngd-pion-s`, memory 82.5 GB
against 83.3.

### There is no do-nothing baseline

Freezing the matrices at initialisation and letting AdamW carry the rest is not
a training configuration: 5.86 at step 100, 5.71 at 150, 23.1 at 500, 1769 at
2000, dead at ~2300. An earlier reading of the 150-step board as "the rotation
mostly hurts" was measuring the transient minimum of that diverging curve and
has been withdrawn in the journal.

### What to do next, in order

1. **`ngd-pion-s` at full length, `eta = 1e-2`, ~10 rtx-hours.** It leads `pion`
   at every horizon from 500 to 3000 and has never been run to 73 242 steps.
   The only full-length NGD run used `S = I` at `eta = 1.0` -- the variant now
   known to be worse, at an `eta` chosen by a method now known to invert. This
   is the run the paper would report and it does not exist.
   **Caveat to state when it is launched:** `eta = 1e-2` was itself chosen at
   150 steps. Consider validating it at 3000 first, which costs 4% of the run.
2. `pion_ablated` at full length, still the isolating baseline for any claim
   about what the preconditioner buys.
3. Shampoo needs `eta` above `1e-1` before any comparison involving it is fair.
4. Harness: abort on a non-finite loss with a clear message. A diverged run
   currently dies inside `linalg.solve` reporting a singular matrix, which
   reads as a numerical bug in the optimizer and is not one.

### Open bugs

* **`ngd_power` is silently inert for `ngd-pion-s`** -- reaches the manifest,
  the hash and the directory name without reaching the optimizer.
* **`pion_ablated` manifests record the un-ablated settings** (`momentum="lie"`,
  `scaling="rms"`, `retraction="trunc"`) because the manifest serialises
  `RunConfig` while `build_optimizers` substitutes `none`/`none`/`cayley` at
  construction. The runs are correct; the record of them is not.
* `powered.py`'s docstring calls `power = 1/2` "Adam, Shampoo". The exponents
  agree arithmetically; the matrices being raised to them do not.
* `basis_congruence` crashed a run at step 41 500. `safe_eigh` now has a
  three-rung fallback, but the long `ngd-pion-s` run has never been attempted
  and this is where it would bite.
* `is_identity` uses `atol = 1e-6`; a flat-spectrum 512x512 fp32 weight misses
  it at 1.073e-06.
* `shampoo-pion` does not approach the inert limit monotonically as `eta -> 0`:
  5.7071 at 1e-12 but 6.164 at 1e-4 (150 steps, `eps = 1e-6`). Unexplained,
  deliberately.

### Decided, do not re-litigate

* **The Fisher self-scaling hypothesis is dead**, by three direct measurements
  (`kappa = 1.8e-3`, `kfac/exact = 0.0128`, `alpha_exact/alpha` 5e-3 to 1e-1).
  `eta` is a tuned learning rate. Do not re-derive `eta* = 2`.
* **150 steps cannot rank these optimizers.** See above. This supersedes the
  older "a single 150-step run resolves about 0.05 in loss", which was true
  about noise and silent about bias.
* **The anchor is accepted** against its pre-registered criteria, deliberately,
  by the user.
* **Pin `--adamw-lr` for any sweep over `lr`.**
* **Never compare two optimizers at a shared `eta`.**
* **Concurrent runs on one node must share a seed** -- identical corpus windows
  warm each other's page cache instead of competing. Measured: five co-located
  arms reach 115229 tok/s while their own first window reads 23666.
* **Pin the node for arrays** (`-w rtx6002,rtx6003`). Left to itself SLURM
  packed eight tasks onto the one loaded node and step 0 took 185 s against 80.
* **Concurrent GPUs cap at 8 per user**, whatever `sacctmgr` reports.
* `shampoo-pion` is deterministic at 150 steps across nodes and days, to four
  decimals. Do not assume the same of the Fisher variants.
