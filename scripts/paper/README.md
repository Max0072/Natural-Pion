# Job scripts behind the paper

Every number in [`docs/paper/FINAL.md`](../../docs/paper/FINAL.md) comes from one of these scripts. Each header
states the hypothesis, the arms, the rule decided before the run and the prediction. The results are in
`runs-record/<family>/` (manifest and log of every run) and are re-tabulated by
`scripts/analysis/final_tables.py`. Submit from the repository root (`logs/` must exist).

Historical note: the scripts were written while the method lived in the probe `scripts/probes/step_norm.py`
(optimizer name `ngd-pion-m` plus `STEP_NORM_*` flags). `natural-pion-fast` is that exact configuration as a
named optimizer (bit-identical, `tests/test_natural_pion.py`, `natural_pion_check.sbatch`); new runs should use it:

    python -m harness.run --optimizer natural-pion-fast --lr <peak angle> ...

| FINAL.md | what | script | runs-record |
|---|---|---|---|
| 1 | Pion lr grid at 3840 and 19 200 steps | `step_norm_long.sbatch` (9, 10, 15-17) | `stepnormlong` |
| 4, 5.3 | Algorithm 1 vs 2; Algorithm 1 seeds at 3000, 15 000 | `step_norm_momseeds`, `step_norm_mom15k`, `step_norm_final` | `stepnormmomseeds`, `stepnormmom15k`, `stepnormfinal` |
| 5.1 | full length: Pion 73 242; ours 56 900 and 73 242 | `step_norm_long.sbatch` (11-14) | `stepnormlong` |
| 5.2 | same wall time, seeds 0-3 | `step_norm_final.sbatch`, `step_norm_long.sbatch` (0-8) | `stepnormfinal`, `stepnormlong` |
| 5.4 | cost per part of the step | `step_norm_prof.sbatch`, `step_norm_combo.sbatch` | `stepnormprof`, `stepnormcombo` |
| 6.1 | natural gradient vs raw generator, with and without momentum | `step_norm_abl.sbatch`, `step_norm_mom.sbatch` | `stepnormabl`, `stepnormmom` |
| 6.2 | how the length is set: spectral, RMS, Fisher, before F^-1 | `step_norm.sbatch`, `step_norm_norm.sbatch`, `step_norm_pre.sbatch`, `step_norm_rmsseeds.sbatch`, `step_norm_cal.sbatch`, `step_norm_seeds.sbatch` | `stepnorm`, `stepnormnorm`, `stepnormpre`, `stepnormrmsseeds`, `stepnormcal`, `stepnormseeds` |
| 6.3 | rejected choices: per-side sigma, attention angles, flat step spectrum, schedules, RProp, AdamW lr | `step_norm_combo`, `step_norm_attn`, `step_norm_power`, `step_norm_abl`, `step_norm_hyper`, `step_norm_wsd`, `step_norm_early`, `step_norm_rprop` | same names |
| 6.4 | efficient form vs exact: t_fac, A and D windows, batching | `step_norm_tfac`, `step_norm_tfacmom`, `step_norm_factors`, `step_norm_combo`, `natural_pion_check` | `stepnormtfac`, `stepnormtfacmom`, `stepnormfactors`, `stepnormcombo` |
| 6.5 | initial spectrum | `step_norm_init.sbatch`, `step_norm_initlong.sbatch` | `stepnorminit`, `stepnorminitlong` |
| - | early clock check, 15 000-step ladder | `step_norm_clock.sbatch`, `step_norm_15k.sbatch` | `stepnormclock`, `stepnorm15k` |

Two Pion references come from earlier work, in `scripts/archive/sbatch/`: Pion at 3000 steps, seeds 0-3
(stage-1 runs) and Pion at 15 000 steps (`ladder15k.sbatch`).

Muon is run from the Natural-Muon repository (`scripts/sbatch/train.sbatch`, job 397404).
