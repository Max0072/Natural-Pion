#!/bin/bash
#SBATCH -A p330
#SBATCH -J grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# A 2x2 in which no pair differs by more than one thing.
#
# Three conclusions on this project have now come from comparisons where
# something other than the variable under test also differed: the anchor gap
# read across mismatched step counts (52 600 against 56 500), the 2.3x read
# across mismatched micro-batches, and very nearly a verdict on micro-batch 512
# read across a changed allocator. Hence: same step count, same methodology,
# expandable_segments on in every cell, every comparison one move on one axis.
#
#              micro-batch 256   micro-batch 512
#   pion             ?                 ?
#   ngd-pion         ?                 ?
#
# Down a column: NGD-Pion's true overhead at that depth, the optimizer being
# the only difference. Along a row: which micro-batch is actually faster. And
# `pion` at 512 against its own 0.463 s/step from job 241338, which was
# measured without expandable_segments, prices the allocator flag itself.
#
# It also locates the unexplained 0.42 s/step. Job 246610 measured the step's
# three components at 0.730 s against a real 1.153, and the same gap appears
# for pion at 256 but not for pion at 512 -- a pattern no single-cause story
# fits. This grid says whether the gap tracks the depth or the optimizer.
#
#   sbatch -p rtx scripts/sbatch/grid.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
W="$DATA_p330/runs/grid-${SLURM_JOB_ID}"
CELLS="$W-cells.txt"
EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"
COMMON="--precision bf16 --data-path $DATA_p330/c4/c4_train.bin \
        --val-path $DATA_p330/c4/c4_val.bin --eval-every 100000 --log-every 50 --no-resume"

echo "node $(hostname)  partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "expandable_segments: on for every cell"
mkdir -p "$(dirname "$CELLS")"

for OPT in pion ngd-pion; do
  for MB in 256 512; do
    echo
    echo "=== $OPT at micro-batch $MB ==="
    if ! $EXEC python -m harness.run $COMMON --optimizer "$OPT" --micro-batch "$MB" \
           --max-steps 200 --out-dir "$W/$OPT-$MB" > "$W-$OPT-$MB.log" 2>&1; then
      echo "    FAILED: $(grep -oE 'CUDA out of memory' "$W-$OPT-$MB.log" | head -1)"
      tail -3 "$W-$OPT-$MB.log" | sed 's/^/      /'
      continue
    fi
    $EXEC python -c '
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*/log.jsonl") for l in open(f)]
w = sorted(r["tokens_per_sec_window"] for r in rows if r.get("tokens_per_sec_window"))
if w:
    tps = w[len(w) // 2]
    peak = max(r["peak_gb"] for r in rows if "peak_gb" in r)
    print(f"    tokens/s {tps:>9,}   s/step {131072/tps:.3f}   peak {peak:>6.2f} GB"
          f"   9.6B run {9.6e9/tps/3600:>5.1f} h")
    with open(sys.argv[4], "a") as fh:
        fh.write(f"{sys.argv[2]} {sys.argv[3]} {131072/tps:.4f} {tps} {peak}\n")
' "$W/$OPT-$MB" "$OPT" "$MB" "$CELLS"
  done
done

echo
echo "=== the grid ==="
$EXEC python -c '
import sys
cells = {}
try:
    for line in open(sys.argv[1]):
        opt, mb, sstep, tps, peak = line.split()
        cells[(opt, int(mb))] = (float(sstep), int(tps), float(peak))
except FileNotFoundError:
    raise SystemExit("no cells recorded")

print("{:>10}{:>12}{:>12}".format("s/step", "mb 256", "mb 512"))
for opt in ("pion", "ngd-pion"):
    row = "".join(f"{cells[(opt, mb)][0]:>12.3f}" if (opt, mb) in cells else f"{chr(45)*2:>12}"
                  for mb in (256, 512))
    print(f"{opt:>10}{row}")

print()
for mb in (256, 512):
    a, b = cells.get(("pion", mb)), cells.get(("ngd-pion", mb))
    if a and b:
        print(f"  NGD overhead at mb {mb}: {b[0]/a[0]:.3f}x  "
              f"({a[0]:.3f} -> {b[0]:.3f} s/step)  +{100*(b[0]/a[0]-1):.0f}%")
print()
for opt in ("pion", "ngd-pion"):
    a, b = cells.get((opt, 256)), cells.get((opt, 512))
    if a and b:
        print(f"  {opt}: mb256 {a[0]:.3f} vs mb512 {b[0]:.3f} -> "
              f"prefer mb {512 if b[0] < a[0] else 256}")

ref = cells.get(("pion", 512))
if ref:
    print(f"\n  allocator flag: pion mb512 here {ref[0]:.3f} s/step against 0.463 "
          f"without it in 241338 -> {ref[0]/0.463:.3f}x")
print()
for (opt, mb), (sstep, tps, peak) in sorted(cells.items()):
    print(f"  {opt:>9} mb{mb}: {9.6e9/tps/3600:>5.1f} h for 9.6B tokens, peak {peak:.2f} GB")
' "$CELLS"

rm -rf "$W" "$W"-*.log "$CELLS"
