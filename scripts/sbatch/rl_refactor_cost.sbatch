#!/bin/bash
#SBATCH -A p330
#SBATCH -J rlrefcost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Cost of one refactor (congruence eigh) and one Cayley step at Qwen3-1.7B shapes, one GPU (RLVR_PLAN 3.2).
#   sbatch -p rtx -x rtx6003,rtx6004 scripts/sbatch/rl_refactor_cost.sbatch     # and -p b200 when one frees
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/rl_refactor_cost.py
