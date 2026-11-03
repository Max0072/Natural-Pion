#!/bin/bash
#SBATCH -A p330
#SBATCH -J shsweep
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The rest of the `eta` bracket for `shampoo-pion`, one arm per GPU.
#
# `eta = 1e-2` is already done (job 276154, val 5.9795), so this is the other
# five points of the grid in `shampoo.sbatch`.
#
# WHY AN ARRAY ON ONE NODE, and it is not arbitrary. Every arm runs `seed 0`,
# so they read *identical* corpus windows. Co-located, the first arm to touch a
# window warms the page cache and the other four are served from it; spread
# across three nodes the same bytes would be read three times. The measured
# hazard this project already paid for is the opposite arrangement --
# concurrent runs with *different* seeds read different windows and halve each
# other's IOPS, 18.7 s/step for two against 4.4 for one.
#
# WHY `--shampoo-plane-every 0`. Job 276154 carried the plane diagnostic and
# its `svdvals` hit a cusolver convergence failure, falling back to an exact
# method; the plane data is already collected and does not need repeating, and
# leaving it on would confound the throughput reading this run is also for.
#
# WHAT ACCEPTABLE SPEED MEANS HERE, stated before the numbers. A step is
# 131072 tokens on a 60M model, roughly 47 TFLOP forward and backward, so the
# compute-bound floor on this card is about 1 s/step -- and that is exactly
# what `ngd-pion-s` reaches once its corpus windows are in page cache
# (134000 tokens/s). Its *first* arm reads 25629, a factor of 5.2 on identical
# code, so a cold reading measures the filesystem and not the optimizer.
# Job 276154's 18308 was cold, with the diagnostic on, next to a co-tenant.
# The number to read here is the LAST window of each arm, not the first.
#
#   sbatch -p rtx -w rtx6002 scripts/sbatch/shampoo_sweep.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/shampoo"

# Overridable so a single point can be re-run under the array's own conditions
# without a throwaway script. Submit a subset with a matching `--array`, e.g.
#   ETAS="1e-2" sbatch -p rtx -w rtx6002 --array=0-0 scripts/sbatch/shampoo_sweep.sbatch
read -r -a ETAS <<< "${ETAS:-3e-4 1e-3 3e-3 3e-2 1e-1}"
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}  eta=$ETA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer shampoo-pion --lr "$ETA" \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --shampoo-plane-every 0 \
        --max-steps 150 --log-every 30 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
