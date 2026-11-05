# What each probe asks

| probe | writes (runs-record/probes/) | used for |
|---|---|---|
| `paper_checks.py` (+ `.sbatch`) | `paper_checks*.json` | fp32 vs TF32 drift, Newton-Schulz error by angle, warm power iteration, K-FAC against the exact Fisher on a checkpoint |
| `ns_diag.py` (+ `.sbatch`) | `ns_diag_*.json` | step-by-step orthogonality of each retraction over 200 training steps |
| `ns_rounding.py` | `ns_rounding.json` | rounding of the exact solve, standard and correction-form Newton-Schulz on random rotations |
| `retraction_bias.py` | `retraction_bias.json` | the bias tr(R^T R - I)/n of each retraction |
| `retraction_drift.py` | `retraction_drift.json` | spectrum drift of trained checkpoints, one process per checkpoint |
| `optimizer_step_bench.py` (+ `.sbatch`) | `optimizer_step_bench.json` | the optimizer step alone, by part (Table 2b) |
| `pion_angles.py` (+ `.sbatch`) | `pion_angles.json` | the rotation angles Pion takes |
| `rl_refactor_cost.py` | `rl_refactor_cost*.json` | refresh and Cayley cost at Qwen3-1.7B shapes |
| `refresh_bench.py`, `kernel_block.py`, `async_compile_check.py`, `stiefel_bench.py` (+ `.sbatch`) | their own JSON | speed investigations for Fast, Turbo and Stiefel |

The earlier probes of the NGD-Pion work (alpha, floor, trust region, S, step cost, `step_norm.py`) were removed on
2026-11-05; they are in git history, and their findings in `docs/history/`.
