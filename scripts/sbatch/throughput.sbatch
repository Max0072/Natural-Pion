#!/bin/bash
#SBATCH -A p330
#SBATCH -J tput
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Step 1. What a run costs, in the configuration it will actually run in, and
# which micro-batch fits. Both matter: the head materialises vocab-by-tokens
# logits, so memory is not obvious from 58M parameters, and micro_batch decides
# how many of a step's 131072 tokens the covariance is shown.
#
#   sbatch -p rtx  scripts/sbatch/throughput.sbatch
#   sbatch -p b200 scripts/sbatch/throughput.sbatch
#   OPT=ngd sbatch -p rtx scripts/sbatch/throughput.sbatch
#
# Log every 50 steps, as a real run does, not more often. Writing a line costs
# about 1.5 s against an NFS flush, so at --log-every 4 the window is 2 s of
# work and 1.5 s of measuring, and this reported 149,697 tokens/s where the
# anchor runs measure 268,590 by wall clock. The instrument was the difference.
#
# Measured 2026-08-24 on rtx with pion and bf16: micro-batch 512 fits at 83.3 GB
# of 97.9, and a 9.6B run takes 9.9 h -- inside the 24 h cap with room to
# spare, so a full run needs no requeue.
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
OPT="${OPT:-pion}"
W="$DATA_p330/runs/tput-${SLURM_JOB_ID}"
RUN="apptainer exec --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF python -m harness.run"
COMMON="--optimizer $OPT --precision bf16 --data-path $DATA_p330/c4/c4_train.bin \
        --val-path $DATA_p330/c4/c4_val.bin --eval-every 100000 --log-every 50 --no-resume"

echo "node $(hostname)  partition ${SLURM_JOB_PARTITION:-?}  optimizer $OPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BEST=0
for MB in 512 256 128; do
  echo "--- micro-batch $MB ---"
  if $RUN $COMMON --micro-batch "$MB" --max-steps 12 --out-dir "$W/mb$MB" > "$W-mb$MB.log" 2>&1; then
    [ "$BEST" -eq 0 ] && BEST=$MB
    echo "    fits"
  else
    echo "    failed: $(grep -oE 'CUDA out of memory' "$W-mb$MB.log" | head -1)"
  fi
done

echo "--- 200 steps at micro-batch $BEST ---"
$RUN $COMMON --micro-batch "$BEST" --max-steps 200 --out-dir "$W/full" 2>&1 | tail -2
apptainer exec --bind "$DATA_p330:$DATA_p330" "$SIF" python - "$W/full" <<'PY'
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*/log.jsonl") for l in open(f)]
w = sorted(r["tokens_per_sec_window"] for r in rows if r.get("tokens_per_sec_window"))
p = [r["peak_gb"] for r in rows if "peak_gb" in r]
if w:
    tps = w[len(w) // 2]
    print(f"    tokens/s (window, median of {len(w)}): {tps:,}")
    print(f"    peak GB: {max(p) if p else '?'}")
    print(f"    a 9.6B run: {9.6e9 / tps / 3600:.1f} h")
PY
rm -rf "$W" "$W"-mb*.log
