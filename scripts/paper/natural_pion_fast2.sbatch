#!/bin/bash
#SBATCH -A p330
#SBATCH -J npfast2
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The current conservative fast form (2026-11-04): one A per input, Cayley by Newton-Schulz in Z - I, power iteration
# batched (c on the device), bf16 statistics inputs. Seed 0, RTX, as the paper's 3000/15000-step references
# (exact solve, fp32 statistics, loop: 3.7242 / 3.4720, 0.596 s/step).
# PREDICTION: val within 0.010 of the references; s/step about 0.48 at 3000 steps (optimizer-only bench saving 0.116 s).
#   0  3000 steps, angle 0.2    -t 01:00:00
#   1  15000 steps, angle 0.05  -t 03:30:00
#   sbatch -x rtx6003,rtx6004 -t 01:00:00 --array=0 scripts/paper/natural_pion_fast2.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
STEPS=(3000 15000); LR=(0.2 0.05); EV=(500 1000)
OUT="$DATA_p330/runs/npfast2/s${STEPS[$T]}"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --np-retraction ns_delta --np-stats-dtype bf16 --lr "${LR[$T]}" \
      --ngd-beta1 0.9 --train-steps "${STEPS[$T]}" --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 \
      --seed 0 --device cuda --log-every 23 --eval-every "${EV[$T]}" --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" \
      --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
