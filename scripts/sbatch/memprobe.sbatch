#!/bin/bash
#SBATCH -A p330
#SBATCH -J memprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:25:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Round two. Job 246607 settled two things and broke on the third:
#
#   * micro-batch 512 fits with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#     and OOMs without it, so the failure was fragmentation, as the budget
#     suggested -- 385 MB of persistent state against 14.6 GB of headroom.
#   * Pion at micro-batch 256 runs 0.869 s/step, against ngd-pion's 1.076. So
#     NGD-Pion costs +24% over Pion at the same micro-batch, and the 2.3x in
#     the headline was mostly the micro-batch, not the optimizer: the model
#     alone is 1.88x slower at 256 than at 512.
#   * observe_cost.py crashed -- Transformer.forward returns (logits, loss) and
#     the probe treated it as logits. Fixed.
#
# So the remaining question is the one that matters: what does ngd-pion
# actually cost at micro-batch 512, where observe sees twice the tokens in its
# single call per step and therefore gets *more* expensive, not less.
#
#   sbatch -p rtx scripts/sbatch/memprobe.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
W="$DATA_p330/runs/memprobe-${SLURM_JOB_ID}"
ES="--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
BIND="--nv --bind $DATA_p330:$DATA_p330 --pwd $REPO"
EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO $ES $BIND $SIF"
COMMON="--precision bf16 --data-path $DATA_p330/c4/c4_train.bin \
        --val-path $DATA_p330/c4/c4_val.bin --eval-every 100000 --log-every 50 --no-resume"

echo "node $(hostname)  partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for MB in 256 512; do
  echo
  echo "=== where the step goes at micro-batch $MB ==="
  $EXEC python scripts/probes/observe_cost.py --micro-batch "$MB"
done

echo
echo "=== ngd-pion at micro-batch 512, 200 steps, expandable_segments on ==="
$EXEC python -m harness.run $COMMON --optimizer ngd-pion --micro-batch 512 \
      --max-steps 200 --out-dir "$W/full" 2>&1 | tail -2
$EXEC python - "$W/full" <<'PY'
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*/log.jsonl") for l in open(f)]
w = sorted(r["tokens_per_sec_window"] for r in rows if r.get("tokens_per_sec_window"))
if w:
    tps = w[len(w) // 2]
    print(f"    tokens/s (window, median of {len(w)}): {tps:,}")
    print(f"    s/step: {131072/tps:.3f}")
    print(f"    peak GB: {max(r['peak_gb'] for r in rows if 'peak_gb' in r)}")
    print(f"    a 9.6B run: {9.6e9 / tps / 3600:.1f} h")
    print(f"    against Pion at 512 (0.463 s/step): {131072/tps/0.463:.2f}x")
PY

rm -rf "$W"
