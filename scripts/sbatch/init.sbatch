#!/bin/bash
#SBATCH -A p330
#SBATCH -J init
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does the frozen spectrum want to be flat?
#
# Pion and NGD-Pion rotate, so a rotational weight's singular values are fixed
# at initialisation and never move again. The choice is therefore permanent,
# which is why it is worth a run of its own rather than a default.
#
# `AGENTS.md` records the decision as "non-orthogonal (Gaussian, std 0.02),
# because a non-trivial spectrum is wanted for expressiveness". That was
# reasoning, not a measurement at a usable horizon: the initialisation sweep
# that touched it compared seven initialisations **at one shared `eta`**, which
# the journal records as measuring which of them happened to suit that `eta`.
#
# TWO `eta` PER INITIALISATION, for that reason. One arm each would repeat the
# error verbatim.
#
# The control is `ngd-pion-s` on the Gaussian initialisation at its own
# optimum: 4.5947 4.2177 4.0866 4.0124 3.9619 3.9184 at 500..3000.
#
#   sbatch -p b200 scripts/sbatch/init.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/init"

#      init:eta
SPECS=(orthogonal:2e-2 orthogonal:6e-2 xavier:2e-2 xavier:6e-2)
IFS=: read -r INIT ETA <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

# `FORCE=1` takes a run directory's lock from a holder that is gone. Set it only
# after checking with `squeue` that the named job is dead.
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  init=$INIT eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" --init "$INIT" \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume $FORCE_FLAG \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
