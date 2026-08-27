# Start here

A single page for picking this up cold. `AGENTS.md` is the settled state of
play, `docs/JOURNAL.md` is the sequence that produced it (newest last, and long
by now), `docs/CLUSTER.md` is the machine, `$DATA_p330/runs/README.md` says
what every run directory holds. This file is only ever *now*.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-27

### In flight

Nothing. The queue is empty.

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
that is the single most important missing measurement. At 150 steps NGD-Pion
beat `pion_ablated` by 6 sd; at full length there is nothing to compare with.

### The diagnosis, which is the reason to keep going

Per-layer rotation angle spans **4 658x to 36 496x within one step**, sustained
for the whole run, the output projections taking a radian while `wq`/`wk` take
a thousandth. The Fisher is block-diagonal, so it equalises curvature *inside* a
layer and sets no scale *between* layers; one scalar `eta` covers all of them.
Pion has `scaling="rms"` for exactly this.

`S = I` is why, and it is worse than a missing feature -- it gets the sign
wrong:

    true S = W^T E[dd^T] W :  ||X|| ~ 1/(||W|| ||delta|| ||x||)
    S = I                  :  ||X|| ~   ||delta||/(||W|| ||x||)

Curvature goes as `delta^2`, so the step should be *larger* where the backward
signal is weaker. We make it larger where the signal is stronger. Confirmed:
the angle correlates with `||delta||` across layers at **0.92 to 0.98**.

That also explains the shape of the deficit -- flat at 0.28 from step 500 to
25 000, which is a misallocated step rather than an accumulating one.

### What to do next, in order

1. **`pion_ablated` at full length.** About 10 rtx-hours. Without it nothing at
   full length is interpretable. Its own `eta` has never been found beyond 150
   steps, where it was 0.5.
2. **`ngd_pion/with_s.py` is written, wired as `ngd-pion-s`, and untested past a
   CPU smoke run.** `eta` does not carry over: `||D||` is 1e-18 to 1e-12 here,
   so `X` grows by that factor and a smoke test at `lr = 1e-6` already turns 146
   radians. Sweep it from scratch, wide and logarithmic.
3. **Momentum.** NGD-Pion has none; Pion has it in the Lie algebra. Not a
   confound to argue away, a missing part of the method.

### Open bugs

* `basis_congruence` crashed a run at step 41 500 -- `linalg.eigh` failing to
  converge on an ill-conditioned pencil. It will bite **more often** under
  `ngd-pion-s`, where the out-side takes the congruence path too.
* `is_identity` uses `atol = 1e-6` and a flat-spectrum 512x512 weight built in
  fp32 misses it at `1.073e-06`. Nothing takes the cheap basis path that
  should. It should also accept a matrix proportional to the identity, which is
  what a flat initialisation gives a non-square layer.

### Decided, do not re-litigate

* **The anchor is accepted** against its pre-registered criteria, deliberately,
  by the user. Level reproduces to 0.7%; the arm gap comes out 1.9x theirs. It
  licenses internal comparisons in this harness and not the quoting of our
  numbers as reproductions of theirs.
* **Pin `--adamw-lr` for any sweep over `lr`.** One rate drove both optimizers
  until 2026-08-26, so every learning-rate conclusion before that date measured
  AdamW. `adamw_lr = 0` still means "follow lr", which is their published
  design, so the anchor is unaffected.
* **Concurrent runs on one node must share a seed.** Different seeds read
  different corpus windows and halve each other's IOPS: 18.7 s/step for two,
  4.4 as soon as one is cancelled. Sweep seeds sequentially.
* **A single 150-step run resolves about 0.05 in loss.** sd is 0.024 over four
  seeds; resolving 0.02 would need 23 seeds per arm.
* **`T_fac` around 25.** 100 is clearly worse; below 25 nothing is resolvable
  and the cost keeps climbing.
* **`alpha` is vestigial when the basis is current.** It sits at 1.000 for every
  `T_fac <= 10`. It measures staleness, not step size, and cannot bound the
  step: on a fresh basis `quad = curv` is an identity.
