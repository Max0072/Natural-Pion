#!/bin/bash
#SBATCH -A p330
#SBATCH -J notrusthead
#SBATCH --array=0-5%6
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Ablation B of the trust-region discussion (journal 2026-09-22, 02:28): how much
# of NGD-Pion's advantage over Pion lives in `alpha = quad/curv`?
#
# WHAT `alpha` IS. Not a trust region (it compares two model quantities built with
# one operator, so it is 1 on a fresh basis by algebra and reads only basis
# staleness), and not rare: at the tuned rate it is below 0.9 on 67% of layer-steps,
# median 0.64, and 0.20 on the value projections. It is the largest per-layer step
# multiplier in the method, the tuned `rot` is fitted around it, and it was never
# removed at the classical setting. `--ngd-trust none` pins it at alpha_max = 1.
# This is an ablation of a part of the method under test, not an ablated Pion
# baseline (ADR 0008 rejected the latter).
#
# THE ONE VARIABLE. Every flag is that of scripts/sbatch/adamw2d_head.sbatch, the
# seed-0 tuned runs it is compared with, plus `--ngd-trust none`; `rot` is swept
# because without alpha the effective step is longer by 1/alpha (about 1.6x
# typical, 5x on wv), so the optimum has to move down. adamw is held at the tuned
# 8e-3 in this first wave and bracketed in the second.
#
# THE GRID, and why it goes down. Previous no-trust sweep (runs/notrust): best cell
# at the LOWER EDGE, rot 1.25e-3 of 1.25e-3..4e-2, at another protocol (3000 steps
# of a 73 242-step schedule, adamw 1e-3), so its optimum was never found and the
# journal recorded that the grid was wrong in scale. With alpha at 6e-3 and the
# ratios above, the optimum here should sit near 1e-3 or below. Six values, a
# factor 2 apart, from 1.25e-4 to 4e-3.
#
# THE RULE, fixed before any of these numbers exist. Seed 0. With V the best val in
# this grid, the with-alpha seed-0 optimum being 3.7954 and seed-0 Pion 3.8412
# (gap 0.0458), the share of the gap to Pion that survives without alpha is
#
#         R = (3.8412 - V) / 0.0458
#
#     R >= 0.8        alpha is not needed for the advantage. It can be dropped or
#                     simplified, and the case for replacing it (option C) is weak.
#     0.2 <= R < 0.8  part of the advantage lives in alpha. Report the share.
#     R < 0.2         the advantage over Pion IS the step-length damping. The
#                     curvature part is unproven and option C becomes the point.
#
# A best cell on the LOWER edge (1.25e-4) or on the upper edge (4e-3) means the
# grid is extended and R is not quoted. Both arms' optima were picked on seed 0, so
# they share a selection bias that largely cancels in R; n = 1 is not cancelled, so
# a second wave brackets adamw and refines rot, and a third runs seeds 1-3 at the
# optimum, paired with the with-alpha and Pion runs of the same seeds.
#
#   sbatch -p rtx scripts/sbatch/notrust_head.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/notrust2d"

ROT=( 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3 )
L="${ROT[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  ngd-pion-s trust=none  rot=$L  adamw=8e-3  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
