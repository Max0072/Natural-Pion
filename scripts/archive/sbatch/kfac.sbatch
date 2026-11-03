#!/bin/bash
#SBATCH -A p330
#SBATCH -J kfac
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# How much of the curvature underestimate is the independence assumption?
#
# Fitting `rho` against `lr` at step 0 (see docs/JOURNAL.md, 2026-08-27) gives
# `kappa = curv / (4 Q) = 1.8e-3`: the true curvature along our own step is
# about 140x what `curv` reports. That `kappa` lumps together the K-FAC
# independence error, the dropped in/out cross term, and the gap between the
# Fisher and the actual loss curvature.
#
# The cross term is already excluded, by Cauchy-Schwarz rather than by
# measurement: `2 E[ab] <= E[a^2] + E[b^2]`, so it cannot take `kappa` below
# 1/2 whatever the data does.
#
# This probe isolates the independence error alone: it compares
# `4 tr(A X^T S X)` against the exact `E_b[(2 u_b^T X x_b)^2]` along `X`, with
# no full covariance and no independence assumption on the right-hand side.
#
# PRE-REGISTERED, written before the run:
#   ratio <~ 0.0036  ->  independence is the whole story
#   ratio ~  1       ->  independence is fine and the gap is Fisher-against-
#                        Hessian, which no better estimate of A, D or S can fix
#
# No checkpoint argument on purpose: `kappa` was measured at step 0 on freshly
# initialised weights, and the probe's default is exactly that.
#
#   sbatch -p rtx --exclude=rtx6001 scripts/sbatch/kfac.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"

echo "node $(hostname)  job ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env DATA_p330="$DATA_p330" \
    --env DATA_BIN="$DATA_p330/c4/c4_train.bin" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/kfac_error.py
