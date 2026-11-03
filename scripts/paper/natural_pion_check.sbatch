#!/bin/bash
#SBATCH -A p330
#SBATCH -J npcheck
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# natural-pion-fast through the harness, first 500 steps of the 3000-step schedule, against the probe run
# stepnormcombo/batch (same flags). Pass: train_loss at every logged step and val at 499 identical.
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --lr 0.2 --ngd-beta1 0.9 --train-steps 3000 --max-steps 500 \
      --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 \
      --eval-every 500 --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
      --out-dir "$DATA_p330/runs/npcheck" --no-resume
