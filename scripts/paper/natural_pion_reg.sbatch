#!/bin/bash
#SBATCH -A p330
#SBATCH -J npreg
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Re-test the Fisher regularisation on the final method (user, 2026-11-03): the relative floor eps = 1e-4 was chosen
# for the earlier alpha method and never re-tested. natural-pion-fast, 3000 steps, theta 0.2, seed 0.
# Reference eps 1e-4, no damping: 3.7242 (stepnormcombo/batch, bit-identical to natural-pion-fast).
#   0-3  floor only, eps 1e-6 / 1e-5 / 1e-3 / 1e-2
#   4-7  floor 1e-6 plus damping delta * max(F eigenvalue), delta 1e-4 / 1e-3 / 1e-2 / 1e-1
# Rule: a setting better than the reference by >= 0.010 replaces it; otherwise the paper says the choice was
# re-tested and does not matter. PREDICTION (low confidence): flat within 0.010 from 1e-5 to 1e-3; 1e-2 and large
# damping worse (they push F^-1 toward the raw generator, which is 0.12 worse with momentum).
#   sbatch -x rtx6003,rtx6004 --array=0-7 scripts/paper/natural_pion_reg.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
EPS=(  1e-6 1e-5 1e-3 1e-2 1e-6 1e-6 1e-6 1e-6 )
DAMP=( 0    0    0    0    1e-4 1e-3 1e-2 1e-1 )
N="eps${EPS[$T]}-d${DAMP[$T]}"; OUT="$DATA_p330/runs/npreg/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --lr 0.2 --ngd-beta1 0.9 --ngd-eps "${EPS[$T]}" \
      --np-damping "${DAMP[$T]}" --train-steps 3000 --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
      --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --adamw-lr 8e-3 \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
