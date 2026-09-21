#!/bin/bash
#SBATCH -A p330
#SBATCH -J ladder15k
#SBATCH --array=0-9%10
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Stage 3 of docs/PLAN.md, hypothesis H4: does the advantage survive length?
#
# WHAT IS KNOWN. At B = 512, both arms tuned, 3000 steps (393M tokens, 4.1% of the
# budget): NGD-Pion is ahead by +0.035 (seeds 1-3, sd 0.002; seed 0 alone +0.046,
# the seed the optima were picked on). With AdamW pinned at 1e-3 the gap was
# +0.024 at 6000 steps (786M tokens) and -0.014 at 73 242 (9.6B, one run each).
# The two pinned points and the tuned one do not sit on one curve, so this
# measures the horizon with the tuning held: 15 000 steps, 1.97B tokens, 20% of
# the budget, both arms given their own (rot, adamw).
#
# THE GRID. A cross around each arm's 3000-step optimum, ten runs, all seed 0
# (one seed for everyone, as the project rule wants; the seed-to-seed sd of val at
# 3000 steps was 0.002-0.004, so one run per cell resolves a gap of 0.01):
#
#     ngd-pion-s  centre rot 6e-3 adamw 8e-3; rot 3e-3, 1.2e-2; adamw 4e-3, 1.6e-2
#     pion        centre rot 1e-3 adamw 8e-3; rot 5e-4, 2e-3;    adamw 4e-3, 1.6e-2
#
# Why a cross with a factor-2 step and not the factor-3 grid of 3000 steps: the
# optimum moves with the horizon (four orderings have flipped between a short run
# and the full length), so both directions of both rates are bracketed, and the
# 3000-step Pion neighbours at factor 3 were 0.13 worse, i.e. the optimum is
# sharp. A cell that lands on the edge of the cross is a grid that has to be
# extended, and it will be reported as such, not quoted.
#
# THE RULE, fixed before any of these numbers exist. Let gap15 be val(best pion
# cell) minus val(best ngd-pion-s cell), each best taken over its own five cells:
#
#     gap15 >= +0.020   the advantage persists -> the full-length pair is justified
#     +0.010 to +0.020  decaying but present   -> the full-length pair is still
#                                                 worth running, expect it small
#     gap15 <  +0.010   inside noise or gone   -> do not spend the ~30 GPU-hours on
#                                                 full length as the headline; the
#                                                 batch-size regime is the paper
#
# Both bests are picked on the same seed, so both carry the same selection bias and
# it largely cancels in the gap. Not cancelled: n = 1 seed, and a coarse grid.
#
# COST. `ngd-pion-s` about 0.9-1.0 s/step, so about 4 h a run; `pion` about
# 0.5 s/step, so about 2.3 h. Ten runs, about 30 GPU-hours. rtx6002 shows 7 of 8
# GPUs free (another user holds one), so the last three `pion` cells start as
# earlier ones end and the whole array finishes in about 6 h. Checkpoints are
# written every ~4 minutes and resubmitting resumes; they take 780 MB each and
# should be deleted once the numbers are read.
#
# WHY rtx6002 AND log_every 23: docs/CLUSTER.md -- every array completed there
# while the other nodes wedged, and 23 is coprime with t_fac = 25 and small enough
# that a wedge shows within a minute. The card UUID is logged.
#
#   sbatch -p rtx scripts/sbatch/ladder15k.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/ladder15k"

#      -------------- ngd-pion-s ---------------  -------------- pion ------------------
OPT=(  ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s  pion pion pion  pion pion   )
ROT=(  6e-3       3e-3       1.2e-2     6e-3       6e-3        1e-3 5e-4 2e-3  1e-3 1e-3   )
ADW=(  8e-3       8e-3       8e-3       4e-3       1.6e-2      8e-3 8e-3 8e-3  4e-3 1.6e-2 )

T=$SLURM_ARRAY_TASK_ID
O="${OPT[$T]}"; L="${ROT[$T]}"; A="${ADW[$T]}"

echo "node $(hostname)  task $T  $O  rot=$L  adamw=$A  batch=512  steps=15000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer "$O" --lr "$L" --adamw-lr "$A" \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 15000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 1000 \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
