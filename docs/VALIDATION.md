# Validation audit, 2026-09-22

What was checked, how, and what came out. This is an audit of the *state of the
project*, not a new experiment: nothing was submitted to the cluster; every
number below was read from `runs/`, the manifests, the code or the test suite.

Notation used throughout, once, so nothing below is shorthand:

* **val** -- validation loss on held-out C4, cross-entropy in nats per token.
  Lower is better. Every val here is read from `log.jsonl` at the last step.
* **rot** -- learning rate of the rotational optimizer (`--lr`): the one that
  moves the 2-D weight matrices Pion and NGD-Pion own (43.5% of the model).
* **adamw** -- learning rate of AdamW (`--adamw-lr`) on everything else:
  embedding, output head, norm gains (56.5% of the model).
* **B** -- batch size in sequences. One sequence is 256 tokens, so B = 512 is
  131 072 tokens per step, which is the published Pion 60M setting. **B = 512
  is "the classical setting"**; 73 242 steps at that batch is the 9.6B-token
  full run. 3000 steps is 393M tokens, 4.1% of it.
* **gap** -- val(Pion) minus val(NGD-Pion). Positive means NGD-Pion is ahead.
* **seed** -- fixes both the weight initialisation and the order of training
  windows (`harness/data.py`: window order is a function of `(seed, epoch)`).

## Verdicts

| question | verdict |
|---|---|
| Does the code do what the spec says? | **Yes, as far as CPU tests can say.** 262 passed, 1 skipped, 33.6 s |
| Is the `a2dhead` head-to-head a clean comparison? | **Clean in setup, not in noise.** Same seed, data, hardware, code; the arms are two different *presets*, see V2 |
| Does NGD-Pion beat Pion at B = 512? | **Seed 0, 3000 steps: yes, gap +0.046. Not established**: one seed per arm and no seed-noise estimate exists at this length (V4) |
| Was the full-length result a tie "inside Pion's run-to-run spread"? | **No. That reading was wrong** (V1). The like-for-like Pion run is 0.014 *ahead* of NGD-Pion, n = 1 each |
| Are the docs current? | **No.** Five stale statements found and listed in V6 |

## What was checked

| check | method | result |
|---|---|---|
| test suite | `pytest -q` in the container, CPU, 4 threads, login node | 262 passed, 1 skipped, 33.6 s. `AGENTS.md` said 204, `README.md` said 140 |
| arms differ only where intended | diffed `manifest.json` of the two optimum cells (`ngd-pion-s` rot 6e-3 / adamw 8e-3, `pion` rot 1e-3 / adamw 8e-3) | only `lr` and `optimizer` differ. Same `tokens_per_step` 131 072, same `total_tokens` 393 216 000, same seed 0, same node `rtx6002`, same torch 2.13.0a0 / CUDA 13.3 |
| code identical across the two commits | `git diff --stat 4df8f48 f9494ca` | one added file, an sbatch script. The two arms ran on identical code |
| data identical across arms | read `harness/data.py` | window order depends on `(seed, epoch)` only, never on the optimizer. Held-out set is `fixed_batches`, documented as not moving between evaluations. **Not verified**: that the eval seed is the same constant in both arms -- read the docstring, not the call site |
| AdamW built identically | read `harness/train.py::_adamw` | same for every arm: betas (0.9, 0.95), decay 0.1 on 2-D, 0 on 1-D. Sweeps pass `--adamw-lr` explicitly |
| grid edges | tabulated every `adamw2d` and `batchscale` run | V3 |
| the eight full-length Pion runs | diffed their manifests | **V1, the main finding** |

## V1. "Second of nine, inside Pion's own spread" is not a valid reading

`docs/RESUME.md` (2026-08-29) and the journal call `ngd-pion-s` at 3.3860 a tie
because it "sits inside `pion`'s own run-to-run spread, second of nine". The
eight Pion values are not replicates. Reading their manifests:

| val | `alternate` | AdamW `beta2` | norm gains decayed | code state | partition |
|---|---|---|---|---|---|
| 3.3719 | False (bilateral) | 0.95 | no | `cf73187` | rtx |
| 3.3866 | True | 0.95 | no | `cf73187` | rtx |
| 3.3937 | False | 0.999 | yes | `6e738bf` | b200 |
| 3.4059 | False | 0.999 | yes | `7c9783f` | rtx |
| 3.4062 | False | 0.999 | yes | `3bade81` | b200 |
| 3.4080 | True | 0.999 | yes | `6e738bf` | b200 |
| 3.4414 | True | 0.999 | yes | `7c9783f` | rtx |
| 3.4432 | True | 0.999 | yes | `3bade81` | b200 |

(`alternate = True` rotates one side per step; `False` rotates both, "bilateral".
`beta2` 0.999 versus 0.95 and decayed versus undecayed norm gains are the two
harness differences the journal of 2026-08-25 measured at about 0.045 of val.)

Only the top two rows ran on the harness state `ngd-pion-s` was run on
(`beta2 = 0.95`, no decay on norms). The other six sit about 0.045 higher
*because of the harness*, not because of noise. The only genuine noise
measurement in the set is the same-configuration pairs re-run on other hardware:
3.4414 / 3.4432 and 3.4059 / 3.4062, i.e. **0.002**, and that is
non-determinism at a fixed seed, not seed-to-seed variation.

So the like-for-like comparison at full length, AdamW pinned at 1e-3 for both:

    ngd-pion-s  rot 0.01   3.3860   (b200)
    pion, bilateral        3.3719   (rtx)     NGD-Pion behind by 0.014
    pion, alternate        3.3866   (rtx)     level

NGD-Pion moves both sides every step, so bilateral is the matching Pion. **At
full length, pinned AdamW, one run each, NGD-Pion is 0.014 behind.** Whether
0.014 is noise is unknown: no seed-to-seed spread exists at full length.

Also open: the Pion runs above are at code state `cf73187`, the NGD run at
`4a45e6a`. The Pion path is *assumed* unchanged between them; not verified.

Consequence for what to write: the record said "tie". The measured statement is
"slightly behind at full length with AdamW untuned, ahead at short horizons".

## V2. What "pion" and "ngd-pion-s" each switch on

The manifest records only `optimizer = pion` or `ngd-pion-s`. The rest is set by
presets in `harness/train.py::build_optimizers` and the `RunConfig` defaults:

| | `pion` (published Pion) | `ngd-pion-s` |
|---|---|---|
| preconditioner | none, raw generator | Fisher `F^-1` with measured backward covariance `S` |
| momentum | Lie-algebra momentum, `beta1 = 0.9` | none |
| step scaling | RMS, `rms = 0.2` | none |
| retraction | truncated exponential, degree 2 | Cayley, exact |
| query weights | rotated per attention head | rotated whole |
| trust region | none | `alpha = min(1, quad/curv)` |
| refactorisation | -- | every `t_fac = 25` steps |

So the `a2dhead` gap of +0.046 is **NGD-Pion against published Pion**, which is
the comparison a reviewer asks for. It is *not* the contribution of `F^-1`:
five things differ at once. The isolating baseline is `pion_ablated` (no
momentum, no scaling, Cayley -- one variable away from `ngd-pion-s`), and it
has never been run at a tuned AdamW nor beyond 150 steps. That is stage 2 of
`docs/PLAN.md`.

A trap seen in the code and worth writing down: `_adamw` takes
`lr = cfg.adamw_lr or cfg.lr`, so `--adamw-lr 0` (the default) silently runs
AdamW at the rotational rate. That is how an earlier session trained AdamW at
lr = 1. All `adamw2d` and `batchscale` runs pass it explicitly and their
manifests confirm the value; anyone writing a new sbatch should check theirs.

## V3. The grid the head-to-head sits on

B = 512, 3000 steps, seed 0, val (best in bold):

    ngd-pion-s, rot 6e-3      adamw 5e-4 3.9444   1e-3 3.8744   2e-3 3.8276
                              4e-3 3.8072   **8e-3 3.7954**   1.6e-2 3.8020
    ngd-pion-s, rot 2e-3      adamw 8e-3 3.8348
    ngd-pion-s, rot 2e-2      adamw 5e-4 4.0241        (never run at high adamw)

    pion, rot 1e-3            adamw 2e-3 3.8814   **8e-3 3.8412**   1.6e-2 3.8444
    pion, rot 3e-4            adamw 8e-3 3.9739
    pion, rot 3e-3            adamw 2e-3 4.0904   8e-3 3.9756   1.6e-2 4.1185

* **adamw is bracketed for both arms** at their best rot: the worse side is
  measured on each. That closes the edge the earlier grids landed on.
* **rot is bracketed at spacing 3x.** For `pion` the neighbours are 0.13 worse,
  so the optimum is near 1e-3 but could sit anywhere in about 5e-4 to 2e-3,
  worth an unknown few hundredths. For `ngd-pion-s` the upper side (2e-2) was
  only run at adamw 5e-4; at the pinned adamw 1e-3 over 6000 steps it was worse
  than 6e-3 (3.7824 against 3.7149), so it is probably bracketed too.
* The B = 2048 head-to-head is **not** closed on the Pion side: at adamw 3.2e-2
  it reached 4.1592, still better than at 1.6e-2 (4.1845), so its optimum lies
  further out and the quoted gap of 0.108 is an upper bound.

V3b. B = 2048, 750 steps (same 393M tokens): `ngd-pion-s` best 4.0514
(rot 6e-4, adamw 8e-3, interior); `pion` best so far 4.1592 (rot 1e-3, adamw
3.2e-2, **on the edge**). Gap 4.1592 - 4.0514 = 0.108, at most.

Absolute numbers, which the gap alone hides: at the same 393M tokens B = 2048
is worse than B = 512 by **0.26** for `ngd-pion-s` (4.0514 against 3.7954) and
by **0.32** for `pion` (4.1592 against 3.8412). The gap grows from 0.046 to at
most 0.108 inside the worse regime; that is a statement about noise, not a
reason to use the larger batch.

Five `pion` B = 512 cells have a log stopped at step 0 (rot 3e-4 at adamw
5e-4 / 1e-3 / 2e-3; rot 1e-3 at 5e-4 / 1e-3; rot 3e-3 at 5e-4). They are
casualties of the node hangs recorded in `docs/CLUSTER.md`, sit on the
uncompetitive low-adamw side, and change no conclusion. They are noted so that
nobody reads the empty cells as results.

## V4. No noise estimate exists at the lengths that matter

What has been measured:

| what | value | where |
|---|---|---|
| same seed, same config, different hardware, full length | 0.002 | `runs/README.md`, `anchor/` vs `b200/` |
| four seeds, one configuration, **150 steps** | sd 0.024 | `runs/seeds/` |
| `alpha 2` against `alpha 3`, four seeds each | not resolved | `runs/a2/` |

Nothing at 3000 steps and nothing at full length. So **+0.046 has no error
bar**. An earlier message of this session quoted "sd about 0.03 between eight
Pion runs" as the scale of that noise; that number came from the eight
non-replicates of V1 and is void.

## V5. The baseline's own status

The anchor (reproduce the published Pion number in this harness) **missed**:
bilateral 3.3805 against 3.3575 (0.69%), alternate 3.3954 against 3.3654
(0.89%), arm gap 0.0149 against 0.0079 (1.9x). The 2026-08-26 journal records
that the pre-registered criteria were not met and that the user accepted the
anchor by override, for internal comparisons only. What it licenses: comparing
arms inside this harness. What it does not: quoting a Pion number from here as
a reproduction of theirs. This belongs in the paper as a stated limitation.

## V6. Documentation that disagrees with the repository

| where | says | actually |
|---|---|---|
| `AGENTS.md` "State of play" | 204 tests, 26 s | 262 passed, 1 skipped, 33.6 s |
| `AGENTS.md`, `README.md` "Running" | `pytest -q # 140 tests, 24 s` | as above |
| `README.md` "Status" | "Nothing has been trained at scale ... The only evidence the method helps is a toy" | fifteen full-length runs exist and a tuned B = 512 head-to-head; corrected in this pass |
| `docs/RESUME.md` (old) | full-length result is a tie, "second of nine" | V1; rewritten |
| `docs/RESUME.md` (old) open bugs | `ngd_power` inert for `ngd-pion-s` | commit `40a7886` (2026-08-30) says it now reaches the measured-`S` variant. **Not re-tested here**; the calibration job `scripts/sbatch/powercal.sbatch` was written but no `runs/powercal` exists, so the exponent has never been measured on the live arm |
| `ALGORITHM.md` "Решения и почему", "Гиперпараметры", "без momentum", `T_fac = 100` | `S = I`, no momentum, `T_fac` 100 | reversed on 2026-08-27 and 2026-08-28 (`S` measured, momentum is an arm, `t_fac` 25). `AGENTS.md` carries the reversals with strikethroughs; **`ALGORITHM.md` does not**. Left unedited: it is the specification and a rewrite deserves its own pass |

## The idea, claim by claim

What the method rests on, with the status of each piece.

| claim | status | evidence |
|---|---|---|
| the Fisher operator on the bivector space is `F(X) = 2(BXC + CXB)` | verified | against 4M-sample Monte Carlo, residual 1.5e-3 (MC noise); Kronecker solve 2.6e-15 |
| the closed-form solve inverts it; descent lemma and sign hold | verified | `ALGORITHM.md` "Что проверено численно", pinned by tests |
| Cayley keeps singular values exactly; Pion's truncated exponential does not | verified | drift over 2000 steps at angle 0.05: 7.1e-15 against 2.4e-4 |
| the spectrum survives on the GPU | **needs a card** | TF32 destroys it by a relative 1.0 unless switched off; `scripts/gpu_smoke.py` checks it, no CPU test can. Not re-run today |
| the step is 96.4% sampling noise | measured | `split_half_step.py`, real model, one model state |
| the Fisher supplies the step scale (`eta* = 2` is derived) | **refuted** | `kappa` 1.8e-3, `kfac/exact` 0.0128 |
| the preconditioned step is the *natural* gradient | **weaker than named** | K-FAC independence gives 0.0128 of the true curvature along the step; what is applied is an input-covariance-whitened rotation, and "natural gradient" over-describes it |
| the measured `S` beats `S = I` | measured | +0.23 to 0.32 at its own `eta` |
| NGD-Pion beats Pion at B = 512 | **seed 0, 3000 steps only** | V1, V3, V4 |
| the advantage grows with batch | **one seed, two protocols agree** | gap 0.046 -> at most 0.108 (tuned AdamW, 393M tokens) and 0.024 -> 0.136 (AdamW pinned, 786M tokens); both inside a worse regime (V3b) |
| the advantage survives to full length | **no evidence, some against** | pinned AdamW: +0.024 at 786M tokens, -0.014 at 9.6B; the two are the only comparable pair and it goes the wrong way |
| `F^-1` is what produces the advantage | **not tested** | needs `pion_ablated` at a tuned AdamW (V2) |

The last row is the one the paper's claim actually depends on, and it is the
only one of the twelve with no measurement at all.

## Not checked

* The GPU spectrum smoke test (needs a node).
* `docs/CLUSTER.md`, and most of the 5470-line journal: read the index and the
  last two entries only.
* That Pion's code path is unchanged between `cf73187` and `4a45e6a`.
* That the two arms use the same validation seed at the call site.
* Whether `ngd_s_power` works end to end on the live arm.
