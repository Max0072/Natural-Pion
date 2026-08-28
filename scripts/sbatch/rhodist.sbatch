#!/bin/bash
#SBATCH -A p330
#SBATCH -J rhodist
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# What does the reduction ratio actually look like?
#
# Every `rho` this project has quoted was sampled at a refactorisation
# boundary: `log_every = 100` is a multiple of `t_fac = 25`, so the logged step
# always has a fresh basis -- `alpha = 1`, the longest step of the cycle, the
# lowest `rho`. The rule in job 297936 adapts every 5 steps and sees every
# phase, which is why its behaviour contradicted the logged numbers.
#
# **`log_every = 97`** is coprime with 25, so the logged rows walk through the
# cycle instead of standing on one point of it, and `harness.train` now reports
# `rho_med`, `rho_lo`, `rho_hi` over every measurement in the window rather
# than the instantaneous value.
#
# TWO ARMS, BECAUSE ONE CANNOT LOCATE A BAND. The trust region needs to know
# which `rho` marks a good step, and that is where the distributions at a good
# and a bad `eta` separate:
#
#   * `eta = 0.02` -- the swept optimum, val 3.9184 at 3000.
#   * `eta = 0.15` -- where the adaptive rule settled on its own, val ~4.24.
#
# If the two distributions overlap, `rho` does not discriminate and targeting
# it cannot work, whatever the thresholds. That is a real possible outcome and
# it would close the line.
#
# `trust_lr` is OFF here on purpose: this measures the ratio at a fixed rate
# rather than watching a controller chase it.
#
#   sbatch -p rtx scripts/sbatch/rhodist.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/rhodist"

ETAS=(2e-2 1.5e-1)
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  eta=$ETA log_every=97 rho_every=5"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" \
        --rho-every 5 --rho-micro 128 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 97 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
