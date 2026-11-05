# Job scripts behind the paper

Every number in `docs/paper/preprint.md` is computed by `scripts/analysis/paper_numbers.py` from the runs in
`runs-record/` (manifest and log of every run) and from the probe outputs in `runs-record/probes/`. Submit from the
repository root (`logs/` must exist).

| script | what it runs |
|---|---|
| `v2.sbatch` | every Natural Pion run of the paper's second version, one row per run (Sections 3-5: retractions, statistics precision, regularisation, true Fisher, Algorithm 1 and the speed-up chain, the 5.4 ablations, the budget curve, full length) |
| `pion.sbatch` | every Pion run the paper reads, repeated exactly from its manifest (`harness.run --from-manifest`) |
| `collect_npv2.sh` | copies the logs of `v2.sbatch` runs into `runs-record/npv2` and re-renders the paper |
| `natural_pion_{check,fast2,ns,reg,stats}.sbatch` | the first version's Natural Pion runs (query projection whole), superseded by `v2.sbatch` |
| `natural_pion_turbo.sbatch` | the Turbo tier (not in the paper) |

The method is `--optimizer natural-pion-fast --lr <peak angle>`; Algorithm 1 is `--optimizer natural-pion`.
The earlier launchers (`step_norm_*.sbatch` on the probe `step_norm.py`, `scripts/archive/sbatch/`) were removed on
2026-11-05 with the legacy optimizers; they are in git history.
