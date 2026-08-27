#!/bin/bash
#SBATCH -A p330
#SBATCH -J warmup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Is `eta` limited by the initialisation transient?
#
# Job 274008 measured `alpha_exact = quad / (4 E_b[s_b^2])` per layer, the
# curvature the step is not preconditioned by:
#
#     step   median alpha_exact/alpha   curv too small by
#        1              4.70e-03              213x
#       31              1.24e-01                8.1x
#       61              4.06e-02               24.6x
#       91              6.62e-02               15.1x
#      121              7.80e-02               12.8x
#
# The 213x is a transient: by step 31 the Kronecker curvature is only ~13x
# short. Since `eta* = 2 kappa`, the step-1 value predicts `eta* = 9.4e-3`
# against a swept optimum of 1.0e-2 -- a 6% match -- while the steady-state
# value predicts 0.15. One scalar `eta` for the whole run is therefore pinned
# by the first thirty steps, and `warmup_steps` is 0 in every run so far.
#
# PRE-REGISTERED:
#   * If the transient is the binding constraint, the optimum under warmup
#     moves to about 0.15 and beats the 5.5068 that `eta = 1e-2` gives without
#     warmup.
#   * `eta = 1` should be WORSE than 0.15: it is seven times above even the
#     steady-state optimum. It is included because it is worth testing rather
#     than assuming.
#   * If nothing beats 5.5068, the hypothesis is dead and the limit is
#     elsewhere.
#
# THE CONTROL, and it is not optional. `lr_at` warms AdamW on the same
# schedule, so any gain could be AdamW's. Arm B holds `eta` at the old optimum
# and turns warmup on: if B alone recovers most of the improvement, the
# rotational story is not established and the entry must say so.
#
#   sbatch -p rtx scripts/sbatch/warmup.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/warmup"

EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"
COMMON="--optimizer ngd-pion-exact --ngd-t-fac 25 --adamw-lr 1e-3 \
        --precision bf16 --seed 0 --ngd-exact-every 30 --ngd-exact-tokens 4096 \
        --max-steps 150 --log-every 30 --no-resume \
        --data-path $DATA/c4_train.bin --val-path $DATA/c4_val.bin --out-dir $OUT"

echo "node $(hostname)  job ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name --format=csv,noheader

#        eta    warmup   role
run () { echo; echo "=== eta=$1 warmup=$2  ($3) ==="
         $EXEC python -m harness.run $COMMON --lr "$1" --warmup-steps "$2" \
           || echo "    FAILED at eta=$1 warmup=$2 -- continuing"; }

# `ARMS` is a space-separated list of `eta:warmup:label`. The default is the
# whole grid; a subset can be run first and the rest appended later, since each
# arm writes its own directory under `$OUT` and the summary reads all of them.
ARMS="${ARMS:-1e-2:30:B-control-isolates-AdamW 5e-2:30:C 1.5e-1:30:D-predicted-optimum 5e-1:30:E 1.0:30:F-proposal}"
for A in $ARMS; do
  IFS=: read -r ETA WARM LABEL <<< "$A"
  run "$ETA" "$WARM" "$LABEL"
done

echo
echo "=== the grid ==="
$EXEC python -c '
import glob, json, os, sys
rows = []
for m in glob.glob(os.path.join(sys.argv[1], "*", "manifest.json")):
    d = os.path.dirname(m); cfg = json.load(open(m))["config"]
    val = float("nan"); steps = -1
    p = os.path.join(d, "log.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try: r = json.loads(line)
            except ValueError: continue
            if "val_loss" in r: val = r["val_loss"]
            if "train_loss" in r: steps = max(steps, r["step"])
    rows.append((cfg["lr"], cfg["warmup_steps"], steps, val))
print("%9s%9s%7s%10s" % ("eta", "warmup", "steps", "val@150"))
for lr, w, steps, val in sorted(rows):
    print(f"{lr:>9g}{w:>9}{steps:>7}{val:>10.4f}")
ok = [r for r in rows if r[3] == r[3]]
if ok:
    best = min(ok, key=lambda r: r[3])
    print(f"\n  best: eta = {best[0]:g} at {best[3]:.4f}")
    print(f"  baseline, eta = 1e-2 with no warmup: 5.5068")
    print(f"  predicted optimum from the steady-state curvature: eta = 0.15")
' "$OUT"
