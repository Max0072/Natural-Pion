#!/bin/bash
#SBATCH -A p330
#SBATCH -J refbench
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/refresh_bench.py on a NaturalPionFast checkpoint: basis-refresh variants, time and direction error.
#   CKPT=<checkpoint.pt> sbatch --export=ALL -x rtx6004 scripts/probes/refresh_bench.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/refresh_bench.py "$CKPT"
