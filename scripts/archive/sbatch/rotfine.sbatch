#!/bin/bash
#SBATCH -A p330
#SBATCH -J rotfine
#SBATCH --array=0-7%8
#SBATCH -w rtx6004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# LR ablation on the live default arm (ngd-pion-s, power=1, trust=quad_curv):
# the user's actual request ("абляция на прекондиционер, собрать лучший
# результат"), clarified after a first pass mistakenly ran an ablation on the
# Fisher exponent p instead (see docs/JOURNAL.md, 2026-09-23 04:15 entry).
#
# docs/VALIDATION.md V3's rot grid is bracketed only at 3x spacing: rot=2e-3
# (3.8348) < rot=6e-3 (3.7954, best) but rot=2e-2 (4.0241) was only ever run
# at adamw=5e-4, not at the good adamw=8e-3 -- so the upper side is not really
# closed. This grid fills in between 2e-3 and 2e-2 at the already-good
# adamw=8e-3, plus a repeat of 6e-3 itself as a noise check (seed 0, so this
# checks run-to-run determinism/logging consistency, not seed noise).
#
# Same protocol as V3: B=512, 3000 steps, seed 0, adamw=8e-3 fixed,
# ngd-t-fac=25, trust=quad_curv (the live default -- no --ngd-trust flag).
#
#   sbatch -p rtx scripts/sbatch/rotfine.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/rotfine"

ROT=( 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 1e-2 1.5e-2 )

T=$SLURM_ARRAY_TASK_ID
L="${ROT[$T]}"

echo "node $(hostname)  task $T  ngd-pion-s  rot=$L  adamw=8e-3  trust=quad_curv  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 \
        --batch-sequences 512 --micro-batch 256 \
        --train-steps 3000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
