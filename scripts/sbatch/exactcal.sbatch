#!/bin/bash
#SBATCH -A p330
#SBATCH -J exactcal
#SBATCH --array=0-5%6
#SBATCH -w rtx6003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Calibration for candidate C, option A: `trust="exact"`, the per-layer trust
# region already built and half-tested (journal, 2026-08-28, "the exact trust
# region: it works as a mechanism"). `alpha_exact = quad / curv_exact` uses the
# TRUE per-token, per-layer curvature from a backward hook, not the stale-basis
# proxy `quad_curv` (`alpha`) is built from -- the theoretically grounded
# replacement the user asked for, given A and S are trusted. Nearly free
# (1.05 s/step measured against 1.004 without it). Its noise problem (curv_exact
# is heavy-tailed, an arithmetic EMA collapsed alpha to 0) was fixed with a
# geometric-mean (log-space) smoothing, `ngd_exact_beta`, but the fix was never
# re-tested in a real run -- the run directories that showed the mechanism
# converging to parity (job 297594, deficit 0.189 -> 0.015 and still closing at
# step 3000) were deleted before the fix landed.
#
# WHY CALIBRATE FIRST RATHER THAN GUESS eta. The old run found the optimum at
# 10x the quad_curv control's eta, inside the 8-25x predicted from
# alpha_exact/alpha measured beforehand -- but that was before the eta
# convention doubled (2026-08-28, same day) and before today's AdamW tuning, so
# a naive "10x today's 6e-3" is a guess, not a calibration. Instead: read the
# rotation ANGLE (already logged, no extra cost) after a short warm-up and match
# it to the known reference -- the quad_curv config at rot=6e-3, adamw=8e-3
# reaches median angle ~1.15-1.23 by steps 100-300 (seed 1 of job 332045). The
# eta whose angle lands nearest that, at the same step window, is the one to
# center a real grid on.
#
# THE SPREAD. 8-25x of 6e-3 is 4.8e-2 to 1.5e-1; six points bracket it with
# margin on both sides: 1e-2, 3e-2, 6e-2, 1e-1, 2e-1, 4e-1.
#
# exact_beta = 0.9: log-space EMA (the fixed behavior), not the arithmetic one
# that collapsed alpha to 0 at beta=0.99. No prior measurement of 0.9
# specifically; if angle looks pathological, that is itself a finding.
#
# 300 steps, adamw fixed at 8e-3 (today's tuned value, not the 1e-3 the 08-28
# run used). log_every 11, small enough to see the angle settle within the run.
#
#   sbatch -p rtx scripts/sbatch/exactcal.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/exactcal"

ETA=( 1e-2 3e-2 6e-2 1e-1 2e-1 4e-1 )
L="${ETA[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  ngd-pion-s trust=exact  eta=$L  adamw=8e-3  exact_beta=0.9  batch=512  steps=300  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 \
        --ngd-trust exact --ngd-exact-beta 0.9 --ngd-exact-tokens 4096 \
        --batch-sequences 512 --micro-batch 256 \
        --train-steps 300 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 11 --eval-every 100 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
