#!/bin/bash
#SBATCH -A p330
#SBATCH -J seedshead
#SBATCH --array=0-2%1
#SBATCH -w rtx6002
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Stage 1 of docs/PLAN.md, hypothesis H1: is the +0.046 real?
#
# The tuned head-to-head at the classical batch (B = 512, 3000 steps, 393M
# tokens) gave, on seed 0 only:
#
#     ngd-pion-s  rot 6e-3  adamw 8e-3   3.7954
#     pion        rot 1e-3  adamw 8e-3   3.8412        gap +0.046
#
# One seed per arm, and no seed-to-seed spread has ever been measured at this
# length, so +0.046 has no error bar. This adds seeds 1, 2 and 3 for both arms
# at exactly those two configurations. Nothing else changes: the flags below
# are the ones of scripts/sbatch/adamw2d_head.sbatch, and the manifests of the
# new runs must differ from the seed-0 ones in `seed` alone -- checked after
# the fact, not assumed.
#
# WHY ONE TASK PER SEED, TWO GPUS EACH. Concurrent runs on one node must share
# a seed, or they read different corpus windows and starve each other for I/O.
# So the two arms of one seed run side by side (one card each, same windows), and
# `%1` makes the three seeds go one after another. A seed's window order and
# its initialisation both come from --seed, so the pair is paired: the gap of a
# seed is a difference of two runs that saw the same data.
#
# WHY rtx6002, AND WHY log_every 23. docs/CLUSTER.md: every array completed here
# while rtx6001/6003/6004 wedged (memory allocated, card idle, step-0 row then
# silence). 23 is coprime with t_fac = 25 and small enough that a wedge shows
# within a minute. The card UUIDs are logged so a job packed onto a shared card
# is caught in seconds.
#
# Decision rule, fixed before any of these numbers exist (docs/PLAN.md, H1):
# the four paired gaps (seed 0 plus these three) must all be positive with
# mean / standard error > 2.35 (one-sided 5%, 3 degrees of freedom).
#
#   sbatch -p rtx scripts/sbatch/seeds_head.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/seedshead"

SEED=$((SLURM_ARRAY_TASK_ID + 1))
IFS=, read -r G0 G1 _ <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
if [ -z "${G1:-}" ]; then echo "expected two GPUs, got '$CUDA_VISIBLE_DEVICES'"; exit 2; fi

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  seed=$SEED  gpus=$G0,$G1"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader
mkdir -p "$OUT"

run() {   # gpu  optimizer  rot  adamw
    CUDA_VISIBLE_DEVICES="$1" apptainer exec --env PYTHONUNBUFFERED=1 \
        --env PYTHONPATH="$REPO" \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
        python -m harness.run \
            --optimizer "$2" --lr "$3" --adamw-lr "$4" \
            --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
            --train-steps 3000 \
            --ngd-t-fac 25 --precision bf16 --seed "$SEED" \
            --log-every 23 --eval-every 500 --no-resume \
            --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
            --out-dir "$OUT"
}

run "$G0" ngd-pion-s 6e-3 8e-3 > "logs/seedshead-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-ngd.log" 2>&1 &
P0=$!
run "$G1" pion 1e-3 8e-3       > "logs/seedshead-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-pion.log" 2>&1 &
P1=$!

wait "$P0"; R0=$?
wait "$P1"; R1=$?
echo "seed $SEED  ngd-pion-s exit $R0   pion exit $R1"
[ "$R0" -eq 0 ] && [ "$R1" -eq 0 ]
