#!/bin/bash
#SBATCH -A p330
#SBATCH -J tfac1a
#SBATCH --array=0-5
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is the 0.0465 between tfac1 (3.8419) and the with-alpha arm (3.7954) alpha's own?
# (journal 2026-10-30.)
#
# THE ONE VARIABLE. Every flag is that of tfac1.sbatch, t_fac 1 included, except
# `--ngd-trust quad_curv`: alpha on. rho_micro stays 128, trajectory-neutral
# (journal, train.py). Same rot grid, without the t_fac 25 check task.
#
# THE RULE, fixed before the numbers. Va = best val here. Q = (3.8419 - Va) / 0.0465:
#     Q >= 0.8       alpha itself gives the rest, on a basis one step old
#     0.2 <= Q < 0.8 both: report the share
#     Q < 0.2        alpha adds nothing on a fresh basis; the lead needs a stale
#                    basis together with alpha, or is something else
# Seed noise is about 0.005 (seeds 0-3 of the with-alpha arm span 0.0098), so Q is
# good to about 0.1-0.2; this is one seed.
# A best cell on an edge of the grid means extend before quoting Q.
#
# EXPECTED RISK: alpha only shrinks the step, and at age 1 the curvature rise is
# about 2.7x (alpha_sources), so the optimum should sit above tfac1's 2e-3, about
# 5e-3, which is past this grid's 4e-3. The top edge is the likely outcome.
#
# Tasks 6-7 close the top edge (4e-3 was best, 3.7843):
#   sbatch --array=6-7 -x rtx6004 scripts/sbatch/tfac1_alpha.sbatch
#
#   sbatch scripts/sbatch/tfac1_alpha.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/tfac1alpha"

ROT=( 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3 8e-3 1.6e-2 )
TFAC=( 1 1 1 1 1 1 1 1 )
L="${ROT[$SLURM_ARRAY_TASK_ID]}"
T="${TFAC[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  ngd-pion-s trust=quad_curv  t_fac=$T  rot=$L  adamw=8e-3  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust quad_curv \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac "$T" --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
