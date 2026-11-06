#!/bin/bash
#SBATCH -A p330
#SBATCH -J anglechk
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# The realised rotation angle of Natural Pion Fast against its promise theta (scripts/probes/angle_check.py).
#   sbatch -x rtx6004 scripts/probes/angle_check.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
OUT="$DATA_p330/runs/anglecheck"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/angle_check.py "$DATA_p330/c4" "$OUT"
