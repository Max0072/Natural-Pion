#!/bin/bash
#SBATCH -A p330
#SBATCH -J preflight
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Step 0. Checks the operations this project depends on, then the two
# properties a CPU test suite cannot reach: that a run resumes from its
# checkpoint on a card, and that the spectrum of a weight holds still even with
# TF32 switched on deliberately.
#
#   sbatch scripts/sbatch/preflight.sbatch
#   sbatch -p b200 SIF=/some/other.sif scripts/sbatch/preflight.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
RUN="apptainer exec --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF python"

echo "node $(hostname)  partition ${SLURM_JOB_PARTITION:-?}"
$RUN scripts/preflight.py
echo "[preflight exit $?]"
$RUN scripts/gpu_smoke.py
echo "[gpu_smoke exit $?]"
