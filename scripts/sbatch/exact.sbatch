#!/bin/bash
#SBATCH -A p330
#SBATCH -J exact
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# `alpha_exact` on the real model, at the swept optimum.
#
# `alpha = quad/curv` is identically 1 on a fresh basis because `curv` is
# formed with the operator that built `X`. `ngd-pion-exact` measures the same
# quadratic form against the per-token truth and logs it beside `alpha`,
# changing nothing about the step -- `_apply` calls its parent first, so the
# trajectory is bit-identical to `ngd-pion-s`.
#
# `eta = 1e-2` is the optimum swept in job 273026 (val 5.5068, bracketed on
# both sides), so this reads the diagnostic on the configuration we would
# actually report rather than on an arbitrary one.
#
# WHAT IS BEING TESTED, stated before the run:
#   * `alpha_exact` should sit far below 1 -- the toy gave 3.4e-6 to 7.6e-3,
#     and `kfac_error.py` plus the `rho` fit bracket the real model's value at
#     roughly 2e-3 to 1.3e-2.
#   * The toy showed attention and FFN differing by ~1000x. Real or an artefact
#     of a two-layer hidden-64 model? This says which.
#   * If `alpha_exact` comes out near 1, the implementation is wrong, not the
#     theory: `curv_exact` would then be reproducing `curv` rather than
#     measuring against it.
#
#   sbatch -p rtx scripts/sbatch/exact.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/exact"

echo "node $(hostname)  job ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-exact --lr 1e-2 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --ngd-exact-every 30 --ngd-exact-tokens 4096 \
        --max-steps 150 --log-every 30 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
