#!/bin/bash
#SBATCH -A p330
#SBATCH -J optbench
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/optimizer_step_bench.py: s/step of the optimizer alone, Pion / Fast variants / Turbo.
#   sbatch -x rtx6003,rtx6004 scripts/probes/optimizer_step_bench.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/optimizer_step_bench.py
