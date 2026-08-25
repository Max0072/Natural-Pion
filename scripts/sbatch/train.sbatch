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
#   sbatch -p rtx scripts/sbatch/train.sbatch --optimizer ngd-pion --lr 1e-3 --train-steps 9155
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
# Measured on 2026-08-25, jobs 246531/246610/246613: this flag is worth 1.60x
# for ngd-pion and exactly nothing for pion (1.599x and 1.597x in two
# independent cells; 1.000x and 1.007x for pion). NGD-Pion makes many
# varied-size allocations per step -- 1376x1376 temporaries, 56 weights, an
# unbatched _apply -- and the caching allocator cannot serve them from a pool
# shaped by the model's activations. Without it a run silently costs 1.6x.
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
RUNS="${RUNS:-$DATA_p330/runs}"

# -J names the project, so every log file is ngd-pion-<id>.out whatever is
# actually running -- which reads as though the optimizer were always ngd. The
# run's real identity is its directory, named by the configuration hash. Rename
# the job after the fact so the queue and the log file say what they are.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update "jobid=$SLURM_JOB_ID" "name=$(echo "$*" | tr ' ' '-' | cut -c1-40)" 2>/dev/null || true
fi

echo "node        $(hostname)"
echo "cpus        ${SLURM_CPUS_PER_TASK:-?}"
echo "gpu         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo unknown)"
echo "job         ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-unknown}"
echo "args        $*"

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --nv \
    --bind "$REPO:$REPO" --bind "$DATA_p330:$DATA_p330" \
    --pwd "$REPO" \
    "$SIF" \
    python -m harness.run \
        --data-path "$DATA/c4_train.bin" \
        --val-path "$DATA/c4_val.bin" \
        --out-dir "$RUNS" \
        "$@"
