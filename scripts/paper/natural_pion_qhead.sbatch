#!/bin/bash
#SBATCH -A p330
#SBATCH -J npqhead
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Query projection rotated per attention head, as published Pion does, so the two settings match.
# natural-pion-fast, 3000 steps, theta 0.2, seed 0. Reference without the split: 3.7242 (stepnormcombo/batch,
# bit-identical to natural-pion-fast). Rule: |d| < 0.010 -> the split does not matter, keep the results;
# otherwise rerun the main comparisons with it. PREDICTION (medium confidence): |d| < 0.010.
#   sbatch -x rtx6003,rtx6004 scripts/paper/natural_pion_qhead.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --np-split-q-per-head true --lr 0.2 --ngd-beta1 0.9 \
      --train-steps 3000 --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda \
      --log-every 23 --eval-every 500 --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
      --out-dir "$DATA_p330/runs/npqhead"
