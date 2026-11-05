#!/bin/bash
#SBATCH -A p330
#SBATCH -J npv2r
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Sections 3-4 and the flag-only part of 5.4 redone in the paper's setting (user, 2026-11-05): query projection per
# head (always, as in Pion), Natural Pion Fast = ns_delta retraction + bf16 statistics unless the row varies them.
# Algorithm 1 = natural-pion (t_fac 1, statistics every step, exact Cayley, fp32). Rows:
#   TAG | OPTIMIZER | STEPS | ANGLE | SEED | BETA1 | RETRACTION | STATS | NP_KWARGS | EXTRA
#   sbatch -x rtx6003,rtx6004 -t <limit> --array=<rows>%6 scripts/paper/v2_rest.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
F=natural-pion-fast; A=natural-pion
ROWS=(
# 0-5  retraction (3.4, 4.8) and statistics precision (4.7); the ns_delta/bf16 reference is npv2/s3000 seed 0
"ret-cayley|$F|3000|0.2|0|0.9|cayley|bf16||"
"ret-ns|$F|3000|0.2|0|0.9|ns|bf16||"
"ret-trunc|$F|3000|0.2|0|0.9|trunc|bf16||"
"ret-trunc-15k|$F|15000|0.05|0|0.9|trunc|bf16||"
"stats-fp32|$F|3000|0.2|0|0.9|ns_delta|fp32||"
"form46|$F|3000|0.2|0|0.9|cayley|fp32|{\"batch_sigma\":false,\"share_inputs\":false}|"
# 6-13 regularisation (5.4)
"reg-eps1e-6|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-6"
"reg-eps1e-5|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-5"
"reg-eps1e-3|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-3"
"reg-eps1e-2|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-2"
"reg-damp1e-4|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-6 --np-damping 1e-4"
"reg-damp1e-3|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-6 --np-damping 1e-3"
"reg-damp1e-2|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-6 --np-damping 1e-2"
"reg-damp1e-1|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-eps 1e-6 --np-damping 1e-1"
# 14-18 true vs empirical Fisher (3.1 remark): sampled targets at three angles, empirical at the two others
"fisher-mc-0.1|$F|3000|0.1|0|0.9|ns_delta|bf16||--ngd-fisher-mc-every 1"
"fisher-mc-0.2|$F|3000|0.2|0|0.9|ns_delta|bf16||--ngd-fisher-mc-every 1"
"fisher-mc-0.4|$F|3000|0.4|0|0.9|ns_delta|bf16||--ngd-fisher-mc-every 1"
"fisher-emp-0.1|$F|3000|0.1|0|0.9|ns_delta|bf16||"
"fisher-emp-0.4|$F|3000|0.4|0|0.9|ns_delta|bf16||"
# 19-24 Algorithm 1 (4.1, 4.2): momentum grid, no-momentum grid
"alg1-mom-0.05|$A|3000|0.05|0|0.9|cayley|fp32||"
"alg1-mom-0.1|$A|3000|0.1|0|0.9|cayley|fp32||"
"alg1-mom-0.2|$A|3000|0.2|0|0.9|cayley|fp32||"
"alg1-nomom-0.2|$A|3000|0.2|0|0|cayley|fp32||"
"alg1-nomom-0.4|$A|3000|0.4|0|0|cayley|fp32||"
"alg1-nomom-0.8|$A|3000|0.8|0|0|cayley|fp32||"
# 25   Algorithm 1 at 15 000 steps
"alg1-mom-15k|$A|15000|0.1|0|0.9|cayley|fp32||"
# 26-31 amortised refresh (4.2): T_fac 25, without and with momentum
"tf25-nomom-0.4|$A|3000|0.4|0|0|cayley|fp32|{\"t_fac\":25}|"
"tf25-nomom-0.8|$A|3000|0.8|0|0|cayley|fp32|{\"t_fac\":25}|"
"tf25-nomom-1.6|$A|3000|1.6|0|0|cayley|fp32|{\"t_fac\":25}|"
"tf25-mom-0.1|$A|3000|0.1|0|0.9|cayley|fp32|{\"t_fac\":25}|"
"tf25-mom-0.2|$A|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25}|"
"tf25-mom-0.4|$A|3000|0.4|0|0.9|cayley|fp32|{\"t_fac\":25}|"
# 32-35 T_fac 10, seeds 0-3
"tf10-s0|$A|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":10}|"
"tf10-s1|$A|3000|0.2|1|0.9|cayley|fp32|{\"t_fac\":10}|"
"tf10-s2|$A|3000|0.2|2|0.9|cayley|fp32|{\"t_fac\":10}|"
"tf10-s3|$A|3000|0.2|3|0.9|cayley|fp32|{\"t_fac\":10}|"
# 36-38 statistics windows (4.3)
"win-adlast6|$A|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":6,\"beta_a\":0.5}|"
"win-betad09|$A|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"beta_d\":0.9}|"
"win-betad095|$A|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"beta_d\":0.95}|"
)
IFS='|' read -r TAG OPT STEPS LR SEED B1 RET STATS KW EXTRA <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
EV=500; [ "$STEPS" -gt 3000 ] && EV=1000
KWARG=(); [ -n "$KW" ] && KWARG=(--np-kwargs "$KW")
OUT="$DATA_p330/runs/npv2x/$TAG"; mkdir -p "$OUT"
echo "row $SLURM_ARRAY_TASK_ID: $TAG $OPT steps $STEPS angle $LR seed $SEED beta1 $B1 $RET $STATS kw=$KW $EXTRA"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer "$OPT" --np-retraction "$RET" --np-stats-dtype "$STATS" "${KWARG[@]}" $EXTRA \
      --lr "$LR" --ngd-beta1 "$B1" --train-steps "$STEPS" --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
      --precision bf16 --seed "$SEED" --device cuda --log-every 23 --eval-every "$EV" --adamw-lr 8e-3 \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
