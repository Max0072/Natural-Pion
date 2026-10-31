#!/bin/bash
#SBATCH -A p330
#SBATCH -J steptm
#SBATCH --array=0-3%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Can we pass Pion on the clock? t_fac 25 costs 0.84 s a step against 1.75 s, Lie momentum almost nothing. Two against t_fac 25 without momentum, best 3.7798.
# 3000 steps at 0.85 s = 2550 s. In 2550 s Pion does 3800-5300 steps (0.67-0.48 s), which by log interpolation of its finals is val 3.795-3.729.
#     best < 3.729   ahead of Pion on the clock, even at its cheaper step cost
#     best > 3.795   behind
#     between        depends on Pion's real step cost; measure both on one node
# Edge means extend. Seed 0. PREDICTION: between, about 3.75.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_tfacmom.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

ROT=( 0.1 0.2 0.4 0.8 )
T=$SLURM_ARRAY_TASK_ID; P=fisher; L="${ROT[$T]}"
OUT="$DATA_p330/runs/stepnormtfacmom"
mkdir -p "$OUT"

echo "node $(hostname)  task $T  precond=$P  momentum=lie  angle=$L  t_fac=25  adamw=8e-3  batch=512  steps=3000  seed=0"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_PRECOND="$P" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 \
        --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
