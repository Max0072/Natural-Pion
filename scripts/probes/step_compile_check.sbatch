#!/bin/bash
#SBATCH -A p330
#SBATCH -J compchk
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# torch.compile (default, reduce-overhead / CUDA graphs) for the optimizer step of Natural Pion Fast
# (scripts/probes/step_compile_check.py). No training.
#   sbatch -x rtx6004 scripts/probes/step_compile_check.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_compile_check.py
