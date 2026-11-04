#!/bin/bash
#SBATCH -A p330
#SBATCH -J npstats
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Cost of the statistics in natural-pion-fast (user, 2026-11-04), seed 0. Both arms share one A between weights fed
# the same input (q, k, v; gate, up), which is bit-identical to not sharing (tests/test_natural_pion.py).
#   STATS=fp32  grams in fp32 (control: val as stepnormcombo/batch 3.7242 up to GPU nondeterminism; timing of sharing)
#   STATS=bf16  inputs rounded to bf16, fp32 accumulation and fp32 gram (tensor cores; still a PSD gram)
# HYPOTHESIS: the rounding adds E[e e^T] >= 0 of relative size ~(2^-9)^2 = 4e-6, below the floor 1e-4 * lam_max,
# so the Fisher basis barely changes. PREDICTION: bf16 val within 0.010 of fp32 at 3000 steps; step faster by most
# of the statistics cost (estimated 0.04-0.07 s at fp32). Rule: adopt bf16 if val is within 0.010 at 3000 and at
# 15000 steps (task 1, run only after task 0 passes).
#   0  3000 steps, theta 0.2     -t 01:30:00
#   1  15000 steps, theta 0.05   -t 04:30:00
#   STATS=bf16 sbatch --export=ALL -x rtx6003,rtx6004 -t 01:30:00 --array=0 scripts/paper/natural_pion_stats.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
STEPS=(3000 15000); LR=(0.2 0.05); EV=(500 1000)
STATS="${STATS:-fp32}"; OUT="$DATA_p330/runs/${OUTROOT:-npstats_${STATS}}/s${STEPS[$T]}"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --np-stats-dtype "$STATS" --lr "${LR[$T]}" --ngd-beta1 0.9 \
      --train-steps "${STEPS[$T]}" --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 \
      --device cuda --log-every 23 --eval-every "${EV[$T]}" --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" \
      --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
