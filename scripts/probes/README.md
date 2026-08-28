# What each probe asks

Thirty-six one-off scripts, written for a specific question and kept because
re-deriving the question costs more than the file does. Grouped by what they
answer. **Twelve of them have no docstring at all** and are listed as such
rather than guessed at.

Most take a run directory as an argument and read `diagnostics.jsonl` or a
checkpoint; a few need a GPU, and say so in their own docstring.

## Cost and precision

| probe | question | answer, where it is settled |
|---|---|---|
| `step_cost_with_s.py` | what measuring `S` costs, away from the data loader | `opt.step()` 240 ms, `A` 106 ms, `D` 151 ms |
| `step_cost_phases.py` | where inside `opt.step()` the 240 ms goes | retraction 54%, angle 22%, factorisation 12% |
| `step_cost.py`, `bench_step.py`, `observe_cost.py` | earlier versions of the same question | superseded by the two above |
| `tf32.py`, `tf32_cost.py` | is fp32 really fp32 on this card, and what does honest fp32 cost | TF32 moves the spectrum by a relative 1.0 over 200 steps; hence `exact_fp32` |
| `autocast_bug.py` | does autocast turn the covariance bf16 despite `exact_fp32` | -- |

## The Fisher, the floor and the step

| probe | question |
|---|---|
| `kfac_error.py` | how far K-FAC is from the empirical Fisher along the direction we step. **0.0128** -- the whole of the missing factor |
| `split_half_step.py` | how much of the step is sampling noise. **96.4%** on the real model |
| `damping_scaling.py` | does the eigenvalue floor matter; the `eps^-1/2` law |
| `analyse_floor.py`, `real_spectra.py`, `inspect_A.py`, `s_spectrum.py` | what the covariance spectra actually look like |
| `curv_collapse.py`, `curv_shapes.py`, `curv_identity.py` | why `quad/curv` blows up, and across which shapes |
| `negative_eigs.py`, `why_negative.py` | whether the covariance carries negative eigenvalues, and at which precision |
| `angle_vs_conditioning.py`, `qkv_controlled_pair.py` | what predicts the rotation angle; the `wq`/`wk` controlled pair |

## The invariant

| probe | question |
|---|---|
| `spectrum_preserved.py` | does a rotational optimizer freeze the spectrum **on real hardware**, not in an fp64 test |
| `spectrum_drift.py` | does the spectrum need to move, measured under AdamW which leaves it free |
| `ns_cayley.py`, `ns_law.py` | Newton-Schulz against the exact retraction |

## Undocumented

`base_apply.py`, `fast_apply.py`, `cancel.py`, `check_cap.py`, `check_patches.py`,
`drift_small.py`, `old_vs_new.py`, `power_acc.py`, `threshold.py`, `trace.py`,
`curv_identity.py`, `angle_vs_conditioning.py` open straight into imports with no
docstring. They were written against a state of the code that has since moved,
and none is referenced by anything. **Read one before trusting it**, and give it
a docstring if it turns out to still work.
