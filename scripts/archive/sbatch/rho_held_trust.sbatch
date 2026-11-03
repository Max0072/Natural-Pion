#!/bin/bash
#SBATCH -A p330
#SBATCH -J rhoheldtr
#SBATCH --array=0-3%4
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The first real test of candidate C. Journal 2026-09-22: a genuine
# Levenberg-Marquardt / trust-region signal already exists in this codebase
# (`rho`, the realised-vs-predicted loss reduction) and is theoretically the
# right kind of correction -- unlike `alpha = quad/curv`, it actually touches
# the realised loss. It was tried twice before and lost both times
# (`trust_lr` on same-batch rho: -0.20 to -0.32 against a swept eta;
# `damped.py`'s LM rule: the adaptation is worth +0.39-0.40 but the variant
# that carries it also removes the spectral floor, costing 0.72 net). Same-
# batch `rho` is "blind by construction" to the step being too long, since it
# is evaluated on the very batch whose gradient produced the step, and the
# step is 96.4% sampling noise -- exactly what a same-batch measurement cannot
# see. `rho_held`, on an independent batch, has been MEASURED since 2026-08-29
# but never ACTED ON: `harness/train.py` fed `adapt_damping` the same-batch
# value on purpose, "one change at a time".
#
# THE ONE CHANGE, made today: `adapt_damping` now reads `rho_held` when the
# holdout is on, `rho` otherwise (harness/train.py, docs/decisions -- ADR to
# follow). Confirmed by a CPU smoke test: rho_held reaches the optimizer and
# lr_scale moves. This is `ngd-pion-s`, floor intact, nothing replaced --
# exactly what the 2026-08-28 journal entry said was "the experiment that has
# not been run".
#
# THE BAND. `RunConfig.ngd_trust_lr_lo/hi` still default to K-FAC's
# `[0.25, 0.75]`, which the comment beside them already says is wrong here --
# job 297936 grew the step at exactly the good rate and settled 2-4x too high.
# The corrected band from job 298962 was `[0.8, 1.2]` (target rho ~ 1), passed
# explicitly below since the stale default was never fixed.
#
# THE GRID. Two starting rates x {same-batch rho, held-out rho}, adamw fixed
# at the tuned 8e-3, seed 0, 3000 steps. rot=6e-3 is the known-good with-alpha
# optimum (val 3.7954, no trust_lr); rot=3e-3 is a lower start, so the run
# also re-checks the "converges from a range of starts" claim on current code.
# The plain no-trust_lr control at rot=6e-3/adamw=8e-3 already exists on disk
# (adamw2d, val 3.7954) and is not rerun.
#
#     task 0: rot 6e-3   same-batch rho
#     task 1: rot 6e-3   held-out rho
#     task 2: rot 3e-3   same-batch rho
#     task 3: rot 3e-3   held-out rho
#
# `rho_every 5`, `rho_micro 128` -- kfac-jax's cadence, and the combination
# `fisher.sbatch` already ran without incident (no OOM: the failure mode this
# guards against needed rho_micro unset, per the 2026-08-29 journal entry).
#
# WHY THIS MIGHT LOSE ANYWAY, said in advance so a loss is not a surprise: the
# step is 96.4% noise, and rho_held's own realised-loss comparison is itself a
# single noisy sample -- the same noise floor that dominates the raw step may
# also dominate the signal a step-size rule needs to read. If it does, this is
# evidence for the noise account, not against the theory.
#
#   sbatch -p rtx scripts/sbatch/rho_held_trust.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/rhoheld"

ROT=(     6e-3   6e-3   3e-3   3e-3  )
HOLDOUT=( false  true   false  true  )

T=$SLURM_ARRAY_TASK_ID
L="${ROT[$T]}"; HO="${HOLDOUT[$T]}"

echo "node $(hostname)  task $T  ngd-pion-s  rot=$L  adamw=8e-3  rho_holdout=$HO  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 \
        --ngd-trust-lr true --ngd-trust-lr-lo 0.8 --ngd-trust-lr-hi 1.2 \
        --rho-every 5 --rho-micro 128 --rho-holdout "$HO" \
        --batch-sequences 512 --micro-batch 256 \
        --train-steps 3000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
