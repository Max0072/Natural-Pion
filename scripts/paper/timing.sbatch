#!/bin/bash
#SBATCH -A p330
#SBATCH -J timing
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Every step time of the paper from one job on one GPU, one configuration after another (Pion first and last, to see
# the node is steady): 600 steps each of the 3000-step schedule, no evaluation, a log row every 100 steps; seconds per
# step = (wall at step 599 - wall at step 100) / 499. Run last, once the method's form is settled.
#   sbatch -x rtx6004 scripts/paper/timing.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
OUTROOT="$DATA_p330/runs/timing"
F=natural-pion-fast
OFF='"stats_window":null,"beta_a":0.95,"beta_d":0.5,"share_inputs":false,"batch_sigma":false,"share_root":false'
RMS='"length":"rms"'
CONFIGS=(
  "pion-a|pion|1e-3|||"
  "alg1|$F|3e-4|cayley|fp32|{\"t_fac\":1,$OFF,$RMS}"
  "tf25|$F|6e-4|cayley|fp32|{\"t_fac\":25,$OFF,$RMS}"
  "windows|$F|6e-4|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,$RMS}"
  "batched|$F|6e-4|cayley|fp32|{\"share_inputs\":false,\"batch_sigma\":false,$RMS}"
  "fast|$F|6e-4|||{$RMS}"
  "final|$F|6e-4|||{$RMS,\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}"
  "pion-b|pion|1e-3|||"
)
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG OPT LR RET STATS KW <<< "$row"
  ARGS=(--optimizer "$OPT" --lr "$LR")
  [ -n "$RET" ] && ARGS+=(--np-retraction "$RET")
  [ -n "$STATS" ] && ARGS+=(--np-stats-dtype "$STATS")
  [ -n "$KW" ] && ARGS+=(--np-kwargs "$KW")
  [ "$OPT" = pion ] || ARGS+=(--ngd-beta1 0.9)
  echo "== $TAG ${ARGS[*]}"
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python -m harness.run "${ARGS[@]}" --train-steps 3000 --max-steps 600 --no-resume --batch-sequences 512 \
        --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 100 --eval-every 100000 \
        --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUTROOT/$TAG" \
      > "logs/timing-$SLURM_JOB_ID-$TAG.log" 2>&1 || echo "   failed: see logs/timing-$SLURM_JOB_ID-$TAG.log"
done
