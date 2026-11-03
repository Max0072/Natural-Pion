#!/bin/bash
#SBATCH -A p330
#SBATCH -J npns
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Newton-Schulz (2 iterations) in place of the Cayley solve in natural-pion-fast (user, 2026-11-03). Seed 0.
#   0  3000 steps, theta 0.2   (Cayley: 3.7242, 0.596 s/step; stepnormcombo/batch)      -t 01:30:00
#   1  15000 steps, theta 0.05 (Cayley: 3.4720; stepnormfinal/tf25-15k-0.05)            -t 04:30:00
# Rule: adopt if val is within 0.010 of Cayley's at both horizons and the step is >= 0.05 s faster; the trained
# spectrum (scripts/probes/paper_checks.py on the checkpoint) must not move more than 2x Cayley's.
# PREDICTION: val within 0.005; step about 0.10 s faster (retraction 0.157 s -> ~0.05 s).
#   sbatch -x rtx6003,rtx6004 -t 01:30:00 --array=0 scripts/paper/natural_pion_ns.sbatch
#   sbatch -x rtx6003,rtx6004 -t 04:30:00 --array=1 scripts/paper/natural_pion_ns.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
STEPS=(3000 15000); LR=(0.2 0.05); EV=(500 1000)
OUT="$DATA_p330/runs/npns/s${STEPS[$T]}"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --np-retraction ns --lr "${LR[$T]}" --ngd-beta1 0.9 \
      --train-steps "${STEPS[$T]}" --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 \
      --device cuda --log-every 23 --eval-every "${EV[$T]}" --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" \
      --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
