#!/bin/bash
#SBATCH -A p330
#SBATCH -J alphasrc
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/alpha_sources.py: which of W, A, D does alpha's drop come from?
# Flags are those of seeds_head.sbatch for ngd-pion-s, seed 1 (whose
# diagnostics.jsonl is the alpha record of journal 2026-09-22 02:37), cut at
# 1000 steps -- the window where alpha is small and the gap to Pion is made.
# Check: train_loss must match runs/seedshead/ngd-pion-s-lr0.006-s1-* step by step.
#
# HYPOTHESIS, before the numbers: D's EMA (beta_backward 0.5) is rebuilt half
# from every batch, so r_D carries most of 1/alpha - 1, mostly on wv.
# Prediction: median log(r_D) > log(r_W), log(r_A). If r_W dominates, alpha
# reads the weight's own rotation instead, and the noise reading is wrong.
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/alphasrc"
mkdir -p "$OUT"
echo "node $(hostname)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env ALPHA_SOURCES_OUT="$OUT/alpha_sources.jsonl" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/alpha_sources.py \
        --optimizer ngd-pion-s --lr 6e-3 --adamw-lr 8e-3 \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 --max-steps 1000 \
        --ngd-t-fac 25 --precision bf16 --seed 1 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
