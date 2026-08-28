# The job scripts

Every one carries its own reasoning in the header -- what is being tested, what
was pre-registered before it ran, and what the arms are compared against. That
header is the point; the srun line underneath is the least interesting part.

**The generator convention changed on 2026-08-28** and with it the meaning of
`eta` for every NGD variant and for `pion_ablated`: `generators` now returns
`skew(W^T G)` rather than twice it, so an `eta` written before that date means
**half** what the same number means now. `momentum.sbatch`, `crossover.sbatch`
and `full.sbatch` have had their NGD rates doubled so they still run the
experiments they ran. **Every other script in this directory still carries the
old numbers**, deliberately -- they are the record of what was run, not a menu
-- so double the NGD rate before re-running one. `pion` with RMS scaling and
`shampoo-pion` are unaffected; see `ALGORITHM.md` §1 for why.

**As of 2026-08-28 the account reaches only `b200`.** `rtx` and `genoa` refuse
with "Invalid account or account/partition combination". The `sbatch -p rtx`
lines in the older headers are history. See `docs/CLUSTER.md`.

| script | what it runs |
|---|---|
| `full.sbatch` | the full-length runs, 73242 steps, `SPECS` overridable. No `--no-resume`: a wall is likely and resubmitting continues from the checkpoint |
| `crossover.sbatch` | where the rotation starts paying for itself; 3000 steps, `SPECS` overridable |
| `momentum.sbatch` | the `eta` sweep for `ngd-pion-m` |
| `shampoo.sbatch`, `shampoo_sweep.sbatch`, `shampoo_eps.sbatch` | Shampoo's `eta`, then `eps x eta` two-dimensionally |
| `shampoo_floor.sbatch` | the inert control and the do-nothing line; takes `FORCE=1` to break a stale `.run.lock` |
| `pion150.sbatch` | `pion` at 150 steps -- the control that showed the short protocol inverts |
| `exact.sbatch`, `warmup.sbatch` | `alpha_exact` on the real model, and the warmup hypothesis |
| `anchor.sbatch` and the rest | the anchor calibration and older sweeps |

## Two rules the cluster taught, both the hard way

* **Pin the node** (`-w`). Left alone the scheduler packed eight tasks onto the
  one loaded node; step 0 took 185 s against 80 elsewhere and the batch was on
  course to hit its own wall.
* **Concurrent runs must share a seed.** Identical corpus windows warm each
  other's page cache; different ones halve each other's IOPS. Measured: five
  co-located arms reach 115229 tokens/s while their own first window reads
  23666.
