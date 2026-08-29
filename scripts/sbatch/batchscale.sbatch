#!/bin/bash
#SBATCH -A p330
#SBATCH -J batchscale
#SBATCH --array=0-15%8
#SBATCH -w rtx6003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does the advantage grow with batch? The prediction the noise measurement makes.
#
# THE READING BEING TESTED. `F^-1` lengthens the step in the directions of least
# curvature -- that is the whole idea -- and those are exactly the directions the
# data constrains least, so the method takes its longest steps where its estimate
# is worst. Amari's advantage is stated for the deterministic gradient flow. Four
# other explanations were closed by measurement on 2026-08-29 (the spectral
# floor, the empirical Fisher, basis staleness, momentum at length); this is what
# is left, and it is the one the numbers always supported.
#
# THE NOISE IS NOT IRREDUCIBLE. `noise_vs_batch.py` on the 3000-step checkpoint,
# cross-checked against the original probe to within 6%:
#
#     B      256     512    1024    2048    4096
#     signal  .008    .021    .033    .063    .119
#
# Linear in `B`, no floor, no saturation -- exactly the `1/B` law for the
# variance of a mean. Fitted, 10% signal costs 6x the batch and 25% costs 19x.
# And the optimizer's cost is per step rather than per token, so the same move
# takes the step-time ratio from 2.10x to 1.29x at 4x. One lever, both problems.
#
#     PREDICTION  `ngd-pion-s` improves against `pion` as the batch grows.
#                 `pion` orthogonalises its update and should be less sensitive.
#     CONTROL     fixed **token budget**, 786M either way: 6000 steps at 512
#                 sequences against 1500 at 2048. Equal tokens, equal schedule
#                 arc, evaluations at matched token counts.
#     KILL        if the gap does not move with batch, the noise reading is
#                 wrong too and there is nothing left of the mechanism story.
#
# EACH ARM ON ITS OWN GRID, four points spanning ~30x. Comparing two optimizers
# at a shared rate is the error this project has a standing rule against, and a
# grid whose optimum lands on its edge has burned a day here before.
#
# `train_steps` IS SET, NOT `max_steps`. `lr_at` builds the cosine from
# `cfg.train_steps`, so every 3000-step sweep in this repository has been running
# at very nearly constant rate -- 4.1% of the arc. Consistent across arms, so
# those comparisons stand, but at a fixed token budget the two batch arms must
# each complete a whole schedule or the shorter one is simply a different
# experiment.
#
# KNOWN ASYMMETRIES, both stated rather than fixed. The covariance is fed by one
# micro-batch per step, so at 2048 the gradient sees 4x the tokens and `A`, `D`
# do not -- but they are EMAs over many steps and the comment in `config.py`
# already argues one micro-batch is far more than they need. And AdamW stays
# pinned at 1e-3 for both batches, so it is under-tuned at 2048; that penalty
# falls on both optimizers equally, which is what the comparison needs.
#
# `beta_D` stays at the default 0.5 even though 0.9 measured better today, so
# that this varies batch and nothing else.
#
# `micro_batch = 256`, not the usual 512. At 512 the run peaks at 82.58 GB of a
# 96 GB card and eight of them side by side OOM intermittently in the allocator
# (`expandable_segments: memory mapping failed`, job 300471): the B=512 arms
# never reached step 0 and the B=2048 arms stopped after it. The cards were
# verified clean afterwards, so nothing had leaked -- thirteen gigabytes of
# headroom is simply not enough. Why the identically-configured `fisher` grid
# survived on rtx6002 is not explained; 256 removes the question by halving the
# activation peak. The covariance then sees 256 sequences a step instead of 512,
# uniformly across every arm, and `config.py` already argues one micro-batch is
# far more than it needs.
#
#   sbatch -p rtx scripts/sbatch/batchscale.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/batchscale"

#       ------------- ngd-pion-s, B=512 -------------  ---------- ngd-pion-s, B=2048 ----------
OPT=(   ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s    ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s \
        pion       pion       pion       pion          pion       pion       pion       pion       )
BSEQ=(  512        512        512        512           2048       2048       2048       2048 \
        512        512        512        512           2048       2048       2048       2048       )
LR=(    2e-3       6e-3       2e-2       6e-2          6e-3       2e-2       6e-2       1.8e-1 \
        3e-4       1e-3       3e-3       1e-2          1e-3       3e-3       1e-2       3e-2       )

O="${OPT[$SLURM_ARRAY_TASK_ID]}"
B="${BSEQ[$SLURM_ARRAY_TASK_ID]}"
L="${LR[$SLURM_ARRAY_TASK_ID]}"
# 786M tokens either way, and evaluations at matched token counts.
if [ "$B" = "512" ]; then STEPS=6000; EVERY=500; else STEPS=1500; EVERY=125; fi

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}  $O  batch=$B  lr=$L  steps=$STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer "$O" --lr "$L" \
        --batch-sequences "$B" --micro-batch 256 \
        --train-steps "$STEPS" \
        --rho-micro 128 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --log-every 97 --eval-every "$EVERY" --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
