#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepabl
#SBATCH --array=0-13%6
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Two questions on step_norm (journal 2026-10-30). Each varies one thing against job 393882
# (t_fac 1, alpha off, cosine, 3000 steps, seed 0; best 3.7399 at angle 0.4).
#
# A. IS THE FISHER DOING ANYTHING, OR ONLY THE NORMALISATION? Tasks 0-4: X = G, no F^-1,
# same normalisation and code path. Vg = best val. Fisher's gain = Vg - 3.7399.
#     gain >= 0.020   F^-1 matters (4x the seed noise of about 0.005)
#     gain <  0.010   normalisation is the whole effect
#     between         one seed cannot say; run seeds
# Vg against Pion 3.8412 says what spectral normalisation alone is worth against RMS.
# PREDICTION: gain >= 0.020.
#
# B. DOES THE SCHEDULE MATTER? Tasks 5-13: constant, linear, warmcos, each on 3 peak angles;
# the cosine cells exist. Rotational angle only, AdamW stays on its cosine.
#     all three bests within 0.010 of 3.7399   the schedule does not matter
#     else                                     it does; report which
# PREDICTION: constant clearly worse (no decay at the end), linear within 0.010, warmcos
# within 0.010.
# A best cell on an edge means extend before concluding, in A and in B.
#
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_abl.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

PRE=( none none none none none  fisher fisher fisher fisher fisher fisher fisher fisher fisher )
SCH=( cosine cosine cosine cosine cosine  constant constant constant linear linear linear warmcos warmcos warmcos )
ROT=( 0.1 0.2 0.4 0.8 1.6  0.05 0.1 0.2  0.2 0.4 0.8  0.2 0.4 0.8 )
T=$SLURM_ARRAY_TASK_ID; P="${PRE[$T]}"; S="${SCH[$T]}"; L="${ROT[$T]}"
OUT="$DATA_p330/runs/stepnormabl/$P-$S"
mkdir -p "$OUT"

echo "node $(hostname)  task $T  precond=$P  schedule=$S  angle=$L  t_fac=1  adamw=8e-3  batch=512  steps=3000  seed=0"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_PRECOND="$P" --env STEP_NORM_SCHEDULE="$S" --env STEP_NORM_PEAK="$L" --env STEP_NORM_STEPS=3000 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac 1 --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
