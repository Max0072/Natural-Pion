#!/bin/bash
#SBATCH -A p330
#SBATCH -J nores
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does preserving the spectrum substitute for a residual connection?
#
# Without residuals the signal must pass through every weight rather than
# around it, and orthogonal weights preserving the norm through depth --
# dynamical isometry -- is the classical prerequisite for training that way.
# It normally decays as the weights train. **Pion freezes the singular values**,
# so an orthogonal initialisation stays orthogonal for the whole run, exactly,
# as long as the retraction is Cayley. `tests/test_harness.py` pins that: six
# Cayley steps leave the spectrum equal to 1e-5.
#
# THE DESIGN IS A 2x2, init x optimizer, all without residuals. **AdamW is not
# a garnish, it is the control**: the claim is that *spectrum preservation*
# carries the model, and that cannot be read without an optimizer that does not
# preserve it running on the same architecture.
#
#   nores + orthogonal + ngd-pion-s    the hypothesis
#   nores + orthogonal + adamw         does a non-preserving optimizer fail?
#   nores + normal     + ngd-pion-s    separates "no residual" from "orthogonal"
#   nores + normal     + adamw         the far corner
#
# Against, already on disk: residual + normal + ngd-pion-s at its own optimum,
# 4.5947 4.2177 4.0866 4.0124 3.9619 3.9184, and residual + orthogonal +
# ngd-pion-s from job 297760.
#
# TWO THINGS THAT BELONG BESIDE ANY RESULT:
#   * The *weights* stay orthogonal; the block Jacobian does not, because
#     softmax, RoPE and SwiGLU sit in between. This is not full isometry.
#   * Residuals earn their keep at depth and this model has **8 layers**. A
#     negative result here is therefore weak evidence and a positive one is
#     strong. The sharp version of the experiment is 24-32 layers, which is a
#     different model and no longer comparable to the anchor.
#   * One `eta` per arm. A residual-free network may want a different one, so
#     this is directional; the journal records comparing initialisations at a
#     shared `eta` as a mistake made twice.
#
#   sbatch -p b200 scripts/sbatch/nores.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/nores"

#      init:optimizer:eta
SPECS=(orthogonal:ngd-pion-s:2e-2 orthogonal:adamw:1e-3 normal:ngd-pion-s:2e-2 normal:adamw:1e-3)
IFS=: read -r INIT OPT ETA <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  residual=false init=$INIT opt=$OPT eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer "$OPT" --lr "$ETA" --init "$INIT" --residual false \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
