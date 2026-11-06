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
#   5 B^-1/2 by Newton-Schulz   6 T_fac 50   7 all of 1, 2, 4, 5, 6   8 statistics only in the step before a refresh
#   (window 1, no EMA in effect; user 2026-11-05: the EMA over the window averages ~2 steps anyway)
#   10, 11 the new Turbo defaults at 3000 / 15000 steps (-t 03:30:00 for 11); 12 the same at 15000 with C floored
#   (eigh) instead of shifted: the control for the shift on the long horizon (user doubted it)
#   13, 14 Turbo defaults with a warm refresh: a full one every 4th refresh (100 steps) / only the first
#   (task 10 is the reference). PREDICTION: 13 within 0.010 of task 10, 14 within 0.020
#   15, 16 Turbo defaults with A and D as Nystrom sketches of rank 64 / 128 (statistics O(T n k)); reference task 10.
#   PREDICTION: 128 within 0.010, 64 within 0.020
#   9 C shifted + B^-1/2 by Newton-Schulz: one eigendecomposition per side left (without T_fac 50, which costs 0.017)
# PREDICTION: each of 1-6 within 0.010 of task 0; 7 within 0.020.
#   sbatch -x rtx6004 --array=0-7%4 scripts/paper/natural_pion_turbo.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
TAG=(exact fisherbf16 stride4 stride8 cshift nsroot tfac50 all window1 cshift_nsroot default default15k default15k_floor warm4 warmonly sketch64 sketch128)
KW=('{}' '{"fisher_dtype": "bf16"}' '{"stats_stride": 4}' '{"stats_stride": 8}' '{"c_floor": "shift"}'
    '{"inv_root": "ns"}' '{"t_fac": 50}'
    '{"fisher_dtype": "bf16", "stats_stride": 4, "c_floor": "shift", "inv_root": "ns", "t_fac": 50}'
    '{"stats_window": 1}' '{"c_floor": "shift", "inv_root": "ns"}' '__default__' '__default__' '__default__{"c_floor": "eigh"}' '__default__{"warm_refresh": 4}' '__default__{"warm_refresh": 1000000}' '__default__{"stats_sketch": 64}' '__default__{"stats_sketch": 128}')
# tasks 0-9 were run with the earlier exact defaults spelled out by the kwargs above; tasks 10-11 run the
# 2026-11-05 Turbo defaults (bf16 F^-1, shifted C, NS root, window 1) at 3000 and 15000 steps
STEPS=3000; LR=0.2; EV=500
if [[ "${KW[$T]}" == __default__* ]]; then
  KW[$T]="${KW[$T]#__default__}"; KW[$T]="${KW[$T]:-{\}}"
else   # tasks 0-9: the pre-2026-11-05 exact defaults, then the task's own option on top
  KW[$T]=$(python3 -c 'import json,sys; b={"fisher_dtype":"fp32","c_floor":"eigh","inv_root":"eigh","stats_window":6}; b.update(json.loads(sys.argv[1])); print(json.dumps(b))' "${KW[$T]}")
fi
if [[ "${TAG[$T]}" == default15k* ]]; then STEPS=15000; LR=0.05; EV=1000; fi
OUT="$DATA_p330/runs/npturbo/${TAG[$T]}"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-turbo --np-retraction ns_delta --np-stats-dtype bf16 \
      --np-kwargs "${KW[$T]}" --lr "$LR" --ngd-beta1 0.9 --train-steps "$STEPS" --batch-sequences 512 --micro-batch 256 \
      --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every "$EV" --adamw-lr 8e-3 \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
