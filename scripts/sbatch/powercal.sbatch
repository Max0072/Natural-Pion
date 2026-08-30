#!/bin/bash
#SBATCH -A p330
#SBATCH -J powercal
#SBATCH --array=0-3%4
#SBATCH -w rtx6001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# What does the exponent do to the scale of the step? Calibration, not a result.
#
# `ngd_power` was hashed into every run name and passed only to `ngd-pion-pow`,
# which is built on the S=I lineage. So the exponent has never been tried on the
# measured-`S` variant, the one that is 0.23-0.32 better. Fixed today; this
# measures the scale before any grid is chosen.
#
# WHY A CALIBRATION AND NOT A SWEEP. At `power = 1/2` the denominator is
# square-rooted, so the step magnitude moves by orders. Guessing an `eta` range
# for that is how two grids landed on their edges earlier today. The rotation
# angle is logged per layer, so the ratio can simply be read off, and each
# exponent's grid can then be centred where its angle matches the
# known-good `power = 1, eta = 6e-3` configuration.
#
# `trust = "none"` ON ALL FOUR, including `power = 1`. At `power != 1` it is
# refused anyway -- `quad/curv` acquires units of `1/sqrt(d)` and stops being a
# multiplier -- and leaving it on for the `power = 1` arm alone would put
# `alpha` (median 0.34, unaliased) inside the very ratio being measured. One
# variable.
#
# THE SECOND READING IS THE INTERESTING ONE. `powered.py` argues that at
# `power = 1/2` the cross-layer scale fixes itself: with `d ~ ||A|| ||S||` and
# `G_in ~ W delta x` the step comes out independent of the backward signal, the
# activations and the weights. That predicts the per-layer angle spread should
# collapse from the measured 4658x-36496x towards the 2.3-2.8x that
# Shampoo-on-so(n) reaches structurally. That prediction is checkable here, in
# 100 steps, without reference to any loss.
#
# `log_every = 7` is coprime with `t_fac = 25`, so the four refactorisation
# cycles are sampled across all their phases rather than at one. Choosing a
# multiple is what made every `rho` and every `alpha` this project has quoted an
# artefact of the freshest-basis phase.
#
# 100 steps is enough for a scale and is not enough to rank anything. Nothing
# about which exponent trains better may be read from this job.
#
#   sbatch -p rtx scripts/sbatch/powercal.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/powercal"

POW=( 0.25 0.5 0.75 1.0 )
P="${POW[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}  power=$P  eta=6e-3  trust=none  100 steps"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr 6e-3 \
        --ngd-s-power "$P" --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 100 --log-every 7 --eval-every 1000 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
