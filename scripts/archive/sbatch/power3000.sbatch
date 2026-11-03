#!/bin/bash
#SBATCH -A p330
#SBATCH -J power3000
#SBATCH --array=0-8%9
#SBATCH -w rtx6004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Ablation on the preconditioner (ADR 0016): the real 3000-step grid for the
# exponent p in F^-p, at the etas calibrated in powercal2 (job 333929).
#
# CALIBRATION, corrected once already: the first pass compared against the
# quad_curv-damped reference angle (~1.2), which is the wrong target since
# every run here has trust=none (alpha pinned at 1, no damping) -- angle at a
# given eta is dozens of times larger without it. Recalibrated using the
# exact linear scaling of angle in eta at fixed power (neither the generator
# nor the F^-p denominator depends on eta), from powercal2's single measured
# point per power at eta=6e-3, still targeting angle ~1.2:
#
#     power  angle@6e-3   calibrated eta
#     0.25    2631.7        2.7e-6
#     0.5     2077.8        3.5e-6
#     0.75     361.1        2.0e-5
#
# Sanity check: the same method applied to power=1 gives eta ~1.6e-4, close to
# ablation B's empirically-found optimum for trust=none at power=1 (2.5e-4,
# job 332073) -- consistent, gives confidence in the calibration.
#
# power=1 IS NOT RE-RUN HERE: ablation B already gridded it thoroughly (six
# points, 1.25e-4 to 4e-3, optimum 2.5e-4, val 4.1969) and that result is
# reused as this grid's power=1 arm.
#
# power=0 (raw generator, no preconditioning) crashed in calibration (`eigh`
# non-convergence at eta=6e-3, ill-conditioned from an ungoverned step) and
# needs a much smaller eta (the trend across 0.25-1.0 suggests roughly 1e-6 or
# below) -- not attempted here, a separate, careful pass if wanted.
#
# THE GRID: three points per power (the calibrated centre, and x0.4/x2.5
# either side), adamw=8e-3 fixed, trust=none (required by the code for
# power != 1 -- quad/curv is not dimensionless there), seed 0, 3000 steps.
#
# COMPARE AGAINST: ngd-pion-s at power=1/trust=quad_curv (the current default,
# 3.7954); ablation B's power=1/trust=none optimum (4.1969); pion (3.8412).
#
#   sbatch -p rtx scripts/sbatch/power3000.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/power3000"

POW=( 0.25 0.25 0.25 0.5  0.5  0.5  0.75 0.75 0.75 )
ETA=( 1.1e-6 2.7e-6 6.75e-6  1.4e-6 3.5e-6 8.75e-6  8e-6 2.0e-5 5e-5 )

T=$SLURM_ARRAY_TASK_ID
P="${POW[$T]}"; L="${ETA[$T]}"

echo "node $(hostname)  task $T  ngd-pion-s  power=$P  eta=$L  trust=none  adamw=8e-3  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 \
        --ngd-s-power "$P" --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 \
        --train-steps 3000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
