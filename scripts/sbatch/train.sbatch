#!/bin/bash
#SBATCH -A p330
#SBATCH -J ngd-pion
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# 64 CPUs against 8 GPUs on these nodes, so 8 per job is the fair share.
# Without this SLURM hands out one core and the data loader becomes the
# bottleneck long before the GPU does.
#SBATCH --cpus-per-task=8
#SBATCH -p b200
#SBATCH --time=24:00:00
# Long runs default to b200 because every partition caps at 24 hours and a full
# 9.6B run does not fit inside that on the rtx pool. Override with -p for a
# short run: `-p rtx` has 32 GPUs against b200's 16, so it is the better place
# for anything that fits.
#
#   sbatch scripts/sbatch/train.sbatch --anchor bilateral
#   sbatch -p rtx scripts/sbatch/train.sbatch --optimizer ngd --lr 1e-3 --train-steps 9155
#
# A requeued job resumes from its checkpoint automatically -- the run directory
# is named by the configuration hash, so resubmitting the same command picks up
# where it stopped, sampler position included. Pass --no-resume to start over.
#
# One run, one GPU, no distributed anything. A 60M model fits on a single card,
# so N cards are worth far more as N concurrent experiments than as one faster
# experiment, and the sweeps this project needs are wide rather than deep.

# NOTE: SLURM opens the -o/-e paths before this script runs, so `logs/` has to
# exist in the submission directory already. `mkdir -p logs` once, before the
# first sbatch, or the job dies with no output explaining why.
set -euo pipefail

REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
RUNS="${RUNS:-$DATA_p330/runs}"

echo "node        $(hostname)"
echo "cpus        ${SLURM_CPUS_PER_TASK:-?}"
echo "gpu         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo unknown)"
echo "job         ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-unknown}"
echo "args        $*"

apptainer exec --nv \
    --bind "$REPO:$REPO" --bind "$DATA_p330:$DATA_p330" \
    --pwd "$REPO" \
    "$SIF" \
    python -m harness.run \
        --data-path "$DATA/c4_train.bin" \
        --val-path "$DATA/c4_val.bin" \
        --out-dir "$RUNS" \
        "$@"
