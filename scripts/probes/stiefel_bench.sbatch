#!/bin/bash
#SBATCH -A p330
#SBATCH -J stfbench
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/stiefel_bench.py: Turbo vs Stiefel optimizer step, one layer of the 1.3B shapes and 8 layers of 60M.
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    bash -c 'python scripts/probes/stiefel_bench.py 1.3B 1; python scripts/probes/stiefel_bench.py 60M 8'
