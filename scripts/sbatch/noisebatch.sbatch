#!/bin/bash
#SBATCH -A p330
#SBATCH -J noisebatch
#SBATCH -w rtx6003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# The noise-versus-batch curve, before spending any training run on the idea.
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
RUN="${RUN:-$DATA_p330/runs/full/ngd-pion-s-lr0.01-s0-9314af4d3e}"
echo "node $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/noise_vs_batch.py "$RUN" "${BATCHES:-256,512,1024,2048,4096}"
