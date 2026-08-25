#!/bin/bash
#SBATCH -A p330
#SBATCH -J ngd-data
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -p rtx
#SBATCH --time=02:00:00
# Building the corpus is pure CPU -- no --gres, so no GPU is allocated and no
# GPU hours are billed; only cpu TRES at weight 0.0625. It runs on a GPU
# partition only because this account reaches `rtx` and `b200` and nothing else.
#
# `rtx` rather than `b200`, measured: tokenising runs at 708k tokens/s per core
# on the rtx nodes' AMD EPYC 9335 against 288k on b201's Xeon 8592+, and b200's
# small edge in download bandwidth (46 against 41 MB/s) cannot pay for that on a
# CPU-bound job.
#
# 32 shards on 32 cores. The work is 10.2 core-hours, so 32 cores is about 20
# minutes against 40 on 16; the reason not to ask for more is courtesy rather
# than speed, since cores on a GPU node are what let other people use its
# cards. Download is a floor of roughly eight minutes -- 20 GB at the 41 MB/s
# the node saturates at, whatever the number of streams -- and it overlaps the
# tokenising.
#
#   sbatch scripts/sbatch/data.sbatch
#
# Resumable: re-running the same command continues from each shard's marker,
# re-fetches nothing, and does nothing at all once the corpus is complete.
set -euo pipefail

export HF_HUB_DISABLE_XET=1

REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
OUT="${OUT:-$DATA_p330/c4}"
TOKENS="${TOKENS:-1e10}"
SHARDS="${SHARDS:-32}"

echo "node    $(hostname)"
echo "cores   ${SLURM_CPUS_PER_TASK:-?}"
echo "out     $OUT"
echo "tokens  $TOKENS in $SHARDS shards"
date -Is

apptainer exec --env PYTHONUNBUFFERED=1 \
    --bind "$DATA_p330:$DATA_p330" \
    --pwd "$REPO" \
    "$SIF" \
    python scripts/prepare_data.py \
        --out "$OUT" \
        --target-tokens "$TOKENS" \
        --shards "$SHARDS"

date -Is
ls -l "$OUT"/*.bin
