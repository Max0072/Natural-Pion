#!/bin/bash
#SBATCH -A p330
#SBATCH -J powercal2
#SBATCH --array=0-4%5
#SBATCH -w rtx6003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Ablation on the preconditioner itself (ADR 0016): the exponent p in F^-p.
# `power=1` is natural gradient (the current default); `power=0.5` is the
# Adam/Shampoo-style square root; `power=0` is the raw generator, no
# preconditioning at all -- the true zero this project has never had inside
# ngd-pion-s's own code path (op_damped.py / powered.py). Never run on the
# live, measured-S arm before today: `ngd_s_power` reached `ngd-pion-pow`
# only, a dead-end lineage; the original scripts/sbatch/powercal.sbatch
# (2026-08-30) was written for this and never executed.
#
# WHY CALIBRATE FIRST. At power != 1 the step's scale moves by orders (the
# denominator is d^power instead of d), so a shared eta across powers would
# repeat the mistake already made twice this project (two grids landing on
# their own edges from a guessed range). Read the rotation angle after a
# short warm-up and match it to the p=1 reference (median angle ~1.15-1.23 at
# steps 100-300, rot=6e-3, adamw=8e-3 -- seed 1 of job 332045).
#
# trust=none for every power, not just p!=1: alpha is dimensionless only when
# X = F^-1 G holds exactly (powered.py's own docstring), so it is switched off
# uniformly here for a clean, single-variable read of what the exponent alone
# does to the angle.
#
# WHY THIS MATTERS BEYOND CALIBRATION. powered.py's own docstring documents a
# real pathology at power=1 with the empirical (measured-S) Fisher: as the
# model fits, the backward residual delta -> 0, so the S-built denominator
# shrinks and the step can inflate -- measured at step 0, a predicted drop of
# 148 against an actual loss of 10.5, delivering nothing. p=0.5 is predicted
# (not yet measured on this arm) to also collapse the cross-layer angle spread
# structurally, the way Shampoo does, without this shrinkage failure mode. The
# old "power=0.5 loses, 5.824 vs 5.5068" verdict (ngd_pion/README.md) predates
# this calibration and was very likely run at a shared/guessed eta -- void by
# the same logic that voided ngd-pion-op's verdict.
#
# 300 steps, log_every 11 (coprime with t_fac=25), adamw=8e-3 fixed.
#
#   sbatch -p rtx scripts/sbatch/powercal2.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/powercal2"

POW=( 0.0 0.25 0.5 0.75 1.0 )
P="${POW[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  ngd-pion-s  power=$P  eta=6e-3  trust=none  adamw=8e-3  steps=300"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr 6e-3 --adamw-lr 8e-3 \
        --ngd-s-power "$P" --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 \
        --train-steps 300 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 11 --eval-every 100 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
