#!/bin/bash
#SBATCH -A p330
#SBATCH -J memprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:25:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Two questions left by job 246531, which measured ngd-pion at 1.076 s/step and
# 21.9 h for a run, against Pion's 0.463 s and 9.4 h.
#
#   1. Why 2.3x and not the 1.5x predicted from stage timings? The prediction
#      omitted `observe` entirely and borrowed its model term from a Pion run at
#      a different micro-batch. This measures all three terms directly, and
#      measures Pion at micro-batch 256 so the comparison is like for like.
#   2. Can micro-batch 512 be made to fit? Persistent optimizer state is about
#      385 MB and the headroom under Pion is about 14.6 GB, so the failure looks
#      like fragmentation rather than capacity. `expandable_segments` targets
#      exactly that and costs no code change, so it is tried first.
#
#   sbatch -p rtx scripts/sbatch/memprobe.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
W="$DATA_p330/runs/memprobe-${SLURM_JOB_ID}"
EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"
RUN="$EXEC python -m harness.run"
COMMON="--precision bf16 --data-path $DATA_p330/c4/c4_train.bin \
        --val-path $DATA_p330/c4/c4_val.bin --eval-every 100000 --log-every 50 --no-resume"

echo "node $(hostname)  partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "=== 1. where the step goes, all three terms measured ==="
$EXEC python scripts/probes/observe_cost.py --micro-batch 256

echo
echo "=== 2. does micro-batch 512 fit? ==="
for ES in off on; do
  if [ "$ES" = on ]; then
    ENV="--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  else
    ENV=""
  fi
  echo "--- expandable_segments $ES ---"
  if apptainer exec --env PYTHONUNBUFFERED=1 $ENV --nv --bind "$DATA_p330:$DATA_p330" \
       --pwd "$REPO" "$SIF" python -m harness.run $COMMON \
       --optimizer ngd-pion --micro-batch 512 --max-steps 12 \
       --out-dir "$W/es-$ES" > "$W-es-$ES.log" 2>&1; then
    echo "    fits"
  else
    echo "    failed: $(grep -oE 'CUDA out of memory' "$W-es-$ES.log" | head -1)"
    grep -oE 'GiB (reserved|allocated)[^.]*' "$W-es-$ES.log" | head -2 | sed 's/^/      /'
  fi
done

echo
echo "=== 3. Pion at micro-batch 256, the like-for-like baseline ==="
$RUN $COMMON --optimizer pion --micro-batch 256 --max-steps 200 --out-dir "$W/pion256" 2>&1 | tail -2
$EXEC python - "$W/pion256" <<'PY'
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*/log.jsonl") for l in open(f)]
w = sorted(r["tokens_per_sec_window"] for r in rows if r.get("tokens_per_sec_window"))
if w:
    tps = w[len(w) // 2]
    print(f"    tokens/s (window, median of {len(w)}): {tps:,}")
    print(f"    s/step: {131072/tps:.3f}")
    print(f"    peak GB: {max(r['peak_gb'] for r in rows if 'peak_gb' in r)}")
PY

rm -rf "$W" "$W"-es-*.log
