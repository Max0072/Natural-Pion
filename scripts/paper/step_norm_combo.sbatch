#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepcmb
#SBATCH --array=0-11%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Everything that helped, in one run: spectral norm after F^-1, Lie momentum 0.9, t_fac 25, equal angles, A and D only in the last 6 steps
# before a refactor, no curv/quad, no diagnostics. Seed 0, 3000 steps.
#   0 idref   300 steps, FAST off    1 idfast  300 steps, FAST on: train_loss and val must be bit-identical, else FAST is wrong
#   2 cos0.2  cosine, angle 0.2      3 lin0.2  linear, 0.2      4 lin0.1  linear, 0.1
# d = val - 3.7227 (same settings without FAST, A and D windows give 3.7227). Seed spread about 0.006.
#   task 2 must give d = 0 exactly (FAST is identical); only its s/step is new.
#   linear: d <= -0.010 linear stays, |d| < 0.010 undecided, keep cosine, d >= +0.010 cosine.
# PREDICTION: s/step about 0.70 (0.744 less curv/quad and diagnostics). Linear at the best angle d between -0.015 and 0, low confidence;
# it gave -0.0088 without momentum at t_fac 1.
# One variable at a time against task 2 (d as above):
#   5 batch   F^-1 and Cayley batched: |d| < 0.010 and lower s/step is a pass. Predict 0.03-0.08 s saved, low confidence.
#   6,7 tf10-0.2, tf10-0.1  t_fac 10: worth it if the clock improves after the extra refactors. Predict d about -0.010, 0.03-0.06 s dearer.
#   8,9 adw4e-3, adw1.6e-2  AdamW lr, never tuned with this rotation. d <= -0.010 moves it; a win at the edge extends. Predict |d| < 0.010.
#   10,11 sides0.2, sides0.1 each side by its own sigma. d <= -0.010 better. Predict |d| < 0.010, low confidence.
#   sbatch -x rtx6003,rtx6004 --array=0-1 scripts/paper/step_norm_combo.sbatch, then --array=2-4 only if identical
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NAME=( idref idfast cos0.2 lin0.2 lin0.1 batch tf10-0.2 tf10-0.1 adw4e-3 adw1.6e-2 sides0.2 sides0.1 ); N="${NAME[$T]}"
FA=( 0 1 1 1 1 1 1 1 1 1 1 1 ); SC=( cosine cosine cosine linear linear cosine cosine cosine cosine cosine cosine cosine )
L=( 0.2 0.2 0.2 0.2 0.1 0.2 0.2 0.1 0.2 0.2 0.2 0.1 ); MX=( 300 300 "" "" "" "" "" "" "" "" "" "" ); EV=( 100 100 500 500 500 500 500 500 500 500 500 500 )
BT=( 0 0 0 0 0 1 0 0 0 0 0 0 ); SD=( 0 0 0 0 0 0 0 0 0 0 1 1 ); TF=( 25 25 25 25 25 25 10 10 25 25 25 25 ); AW=( 8e-3 8e-3 8e-3 8e-3 8e-3 8e-3 8e-3 8e-3 4e-3 1.6e-2 8e-3 8e-3 )
OUT="$DATA_p330/runs/stepnormcombo/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST="${FA[$T]}" --env STEP_NORM_BATCH="${BT[$T]}" --env STEP_NORM_SIDES="${SD[$T]}" \
    --env STEP_NORM_SCHEDULE="${SC[$T]}" --env STEP_NORM_PEAK="${L[$T]}" --env STEP_NORM_STEPS=3000 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr "${L[$T]}" --adamw-lr "${AW[$T]}" --ngd-trust none \
        --ngd-beta 0.5 --ngd-beta-backward 0.5 \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 3000 --ngd-t-fac "${TF[$T]}" ${MX[$T]:+--max-steps ${MX[$T]}} \
        --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every "${EV[$T]}" --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
