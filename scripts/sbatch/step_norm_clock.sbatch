#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepclk
#SBATCH --array=0-1
#SBATCH -p rtx
#SBATCH -w rtx6006
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Clock check on one node, so the step costs compare. Seed 0.
# Task 0: published Pion, lr 1e-3, 5300 steps: what it does in the 42 min of t_fac 25 with momentum at 3000 (3.7226).
#     Pion val - 3.7226 >= +0.010   we are ahead on the clock at this budget
#     |difference| < 0.010          parity
#     <= -0.010                     behind
# Task 1: t_fac 25, Lie momentum, angle 0.1, 15000 steps, about 3.5 h. gap15 = 3.5254 - val: >= +0.020 persists, +0.010..+0.020 decaying, < +0.010 gone.
# Its clock comparison needs Pion near 26000 steps, which does not exist yet.
# PREDICTION: task 0 parity; task 1 persists.
#   sbatch scripts/sbatch/step_norm_clock.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
OUT="$DATA_p330/runs/stepnormclock"; mkdir -p "$OUT"
echo "node $(hostname) task $T"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
COMMON=( --adamw-lr 8e-3 --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23
         --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT" )
if [ "$T" = 0 ]; then
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python -m harness.run --optimizer pion --lr 1e-3 --train-steps 5300 --eval-every 500 --no-resume "${COMMON[@]}"
else
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_PRECOND=fisher \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr 0.1 --ngd-trust none \
        --train-steps 15000 --ngd-t-fac 25 --eval-every 1000 "${COMMON[@]}"
fi
