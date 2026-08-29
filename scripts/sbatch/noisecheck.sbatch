#!/bin/bash
#SBATCH -A p330
#SBATCH -J noisecheck
#SBATCH -w rtx6003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Two controls on the noise-versus-batch result, which came out at a signal
# fraction of zero and so contradicts the 96.4% (i.e. 3.6% signal) this project
# has on record.
#
#   1. the ORIGINAL probe on the SAME checkpoint. Same formula, same data, no
#      accumulation. If it also reads ~0 the new probe is right and the 3.6%
#      came from somewhere else; if it reads 3.6% the new probe's accumulation
#      is wrong.
#   2. the NEW probe on an EARLY checkpoint (3000 steps, at the swept optimum).
#      The full-length checkpoint is at step 73241, the end of a cosine
#      schedule, where per-step progress has flattened and a vanishing signal
#      is exactly what should be expected. If the signal is present early and
#      gone late, that is a fact about convergence, not about the method.
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
LATE="$DATA_p330/runs/full/ngd-pion-s-lr0.01-s0-9314af4d3e"
EARLY="$DATA_p330/runs/fisher/ngd-pion-s-lr0.006-s0-3cc79b6263"
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader
RUN() { apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" python "$@"; }

echo; echo "=============== 1. original probe, LATE checkpoint (step 73241) ==============="
RUN scripts/probes/split_half_step.py "$LATE" || echo "FAILED"

echo; echo "=============== 2. new probe, EARLY checkpoint (step 3000) ==================="
RUN scripts/probes/noise_vs_batch.py "$EARLY" 256,512,1024,2048,4096 || echo "FAILED"

echo; echo "=============== 3. original probe, EARLY checkpoint ==========================="
RUN scripts/probes/split_half_step.py "$EARLY" || echo "FAILED"
