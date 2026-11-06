#!/bin/bash
#SBATCH -A p330
#SBATCH -J tf32chk
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# TF32 against fp32 for Natural Pion Fast: spectrum drift of the Newton-Schulz Cayley step and optimizer step time
# per tf32 mode (scripts/probes/tf32_check.py). No training.
#   sbatch -x rtx6004 scripts/probes/tf32_check.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/tf32_check.py
