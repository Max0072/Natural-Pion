# Which variant is which

Nine optimizer names are registered in `harness.train.NGD_IMPLEMENTATIONS` and
twenty modules sit in this package. This file says what each one is and whether
it is live, so that picking one does not require reading all of them.

**Read the status column with its caveat.** Almost every "superseded" verdict
below was decided on a **150-step** comparison, and 150 steps was measured on
2026-08-27 to *invert* the ordering of these optimizers: `ngd-pion` beats `pion`
at 150 steps and loses to it by 0.30 at 73242, and for `shampoo-pion` the arm
that is worst at step 500 is best from step 1000. So a dead end here means
"lost a comparison we no longer trust", not "ruled out". Re-testing one at 3000
steps is about fifty minutes.

## The optimizers

**The three live NGD names now share one class**, `unified.NGDPionUnified`, with
the variants as flags rather than as a tower of subclasses. `_apply` had been
copied four times and each copy needed a test to police it. The names are kept
as presets so every sbatch script and every run directory on disk still means
what it says, and `tests/test_unified.py` pins each preset **bit for bit**
against the class it replaced.

| name | class | module | status |
|---|---|---|---|
| `ngd-pion-s` | `NGDPionUnified` | `unified.py` | **live.** Preset: `use_s=True`, no momentum. Every NGD result in the repository |
| `ngd-pion-m` | `NGDPionUnified` | `unified.py` | **live, best number so far.** Preset: `use_s=True` plus `ngd_momentum`. 3.8734 at 3000 steps against `pion`'s best-of-eight 3.9880 |
| `ngd-pion-u` | `NGDPionUnified` | `unified.py` | **live.** Every algorithmic choice from the configuration instead of from the name -- the way to reach a combination nobody subclassed |
| -- | `FastNGDPionS` | `with_s_fast.py` | **oracle, no longer run.** Kept so `tests/test_unified.py` has something to check against |
| -- | `MomentumNGDPionS` | `momentum.py` | **oracle, no longer run.** Same |
| `shampoo-pion` | `ShampooPion` | `shampoo.py` | **live alternative.** Preconditioner from the generators themselves, no hooks. Behind `ngd-pion-s` on loss; the only arm with a cross-layer angle spread near 1 |
| `pion`, `pion_ablated` | `Pion` | `pion_baseline.py` | **the baseline.** `pion_ablated` switches off momentum, RMS scaling and the truncated retraction, and is what an isolating comparison needs |
| `ngd-pion` | `FastNGDPion` | `fast.py` | superseded, and **not** folded into the unified class: it carries `angle_max`, a per-step cap on the rotation that the unified class does not implement. The default is `0`, so nothing has used it, but removing a lever silently is worse than leaving a module |
| `ngd-pion-ref` | `NGDPion` | `optimizer.py` | reference. Unoptimised orchestration, checked against `reference.py` |
| `ngd-pion-s-ref` | `NGDPionS` | `with_s.py` | reference for the `S` family. Now carries the power-iteration angle itself |
| `ngd-pion-exact` | `ExactCurvNGDPionS` | `exact_curv.py` | **diagnostic only.** Measures the curvature the step is not preconditioned by; `_apply` calls its parent first, so the trajectory is unchanged |
| `ngd-pion-op` | `OpDampedNGDPion` | `op_damped.py` | dead end. The floor on the operator rather than on its ingredients. Null result -- **and the null was measured at a shared `eta` across arms**, which the journal records as unable to have shown an effect either way |
| `ngd-pion-pow` | `PoweredNGDPion` | `powered.py` | dead end. `F^-p`; `power = 0.5` measured 5.824 against the `S` arm's 5.5068. **Its docstring calls `power = 1/2` "Adam, Shampoo", which is wrong** -- the exponents agree, the matrices raised to them do not |
| `ngd-pion-damped` | `DampedNGDPionS` | `damped.py` | dead end. Additive Tikhonov with a Levenberg-Marquardt rule on the reduction ratio |

## Everything else

| module | role |
|---|---|
| `reference.py` | the numpy oracle for NGD-Pion. **Never optimise it**; if it and the torch path disagree, it is right until proven otherwise |
| `shampoo_reference.py` | the same role for `shampoo.py` |
| `direction.py` | generators, the Fisher operator, the solve, the trust region |
| `factorization.py` | the bases that diagonalise the operator: identity anchor and congruence |
| `covariance.py` | `A = E[xx^T]`, and it refuses bf16 storage on purpose |
| `linalg.py` | shared primitives: `cayley`, `spectral_norm`, `safe_eigh`, the spectral floor, `exact_fp32` |
| `hooks.py` | feeds `nn.Linear` activations to the optimizer |
| `damp_op.py` | the damping primitive `op_damped.py` is built from |

## The two kinds of flag

Kept apart deliberately, and tested differently.

**Algorithmic** -- `use_s`, `momentum`, `beta1` -- change the trajectory. Each
must reproduce the class it replaces under `torch.equal`, not `allclose`.

**Implementation** -- `retraction`, `angle` -- must not, or must change it by a
stated bound. `angle` feeds the instrument and the Newton-Schulz guard and
nothing in the step, so all three settings must leave the weights identical;
`retraction="ns"` is an approximation, so it is checked against the error bound
`linalg.cayley_newton_schulz` documents and against the spectrum invariant.

Mixing the two in one list is how a knob meant to be free quietly becomes a
variable in a comparison.

## Traps this layout has already sprung

* **A configuration field that reaches the manifest but not the optimizer.**
  `ngd_power` does exactly that for `ngd-pion-s`, so six runs record a
  controlled variable that was not one. `tests/test_shampoo.py` and
  `tests/test_momentum.py` pin the wiring for the two newest optimizers; the
  older ones are unpinned.
* **A manifest that records the configuration rather than the object built from
  it.** `pion_ablated` runs record `momentum="lie"`, `scaling="rms"`,
  `retraction="trunc"` because `build_optimizers` substitutes at construction.
  The runs are correct; their record is not.
* **A subclass whose reason for existing has lapsed.** `FastNGDPionS` was faster
  than its parent until the parent absorbed the change. Its docstring now says
  so, and a test pins the equality.
