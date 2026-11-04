#!/bin/bash
#SBATCH -A p330
#SBATCH -J npturbo
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# NaturalPionTurbo approximations (aggressive tier, in no comparison), 3000 steps, angle 0.2, seed 0, RTX.
# All are upstream of the Cayley gate, so the spectrum is kept; the question is what each costs in loss and saves in
# time. Reference: task 0 (Turbo exact: Newton-Schulz in Z - I, bf16 statistics) and npfast2/s3000 (3.7224, 0.498 s).
#   0 exact   1 F^-1 in bf16   2 statistics every 4th token   3 every 8th token   4 C shifted instead of floored
#   5 B^-1/2 by Newton-Schulz   6 T_fac 50   7 all of 1, 2, 4, 5, 6
# PREDICTION: each of 1-6 within 0.010 of task 0; 7 within 0.020.
#   sbatch -x rtx6003,rtx6004 --array=0-7%4 scripts/paper/natural_pion_turbo.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
TAG=(exact fisherbf16 stride4 stride8 cshift nsroot tfac50 all)
KW=('{}' '{"fisher_dtype": "bf16"}' '{"stats_stride": 4}' '{"stats_stride": 8}' '{"c_floor": "shift"}'
    '{"inv_root": "ns"}' '{"t_fac": 50}'
    '{"fisher_dtype": "bf16", "stats_stride": 4, "c_floor": "shift", "inv_root": "ns", "t_fac": 50}')
OUT="$DATA_p330/runs/npturbo/${TAG[$T]}"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-turbo --np-retraction ns_delta --np-stats-dtype bf16 \
      --np-kwargs "${KW[$T]}" --lr 0.2 --ngd-beta1 0.9 --train-steps 3000 --batch-sequences 512 --micro-batch 256 \
      --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --adamw-lr 8e-3 \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
