#!/bin/bash
#SBATCH -A p330
#SBATCH -J resume
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Does a run survive being killed? Not gracefully stopped -- killed, with
# SIGKILL, the way a node failure kills it: no chance to finish the step, close
# the file, or write anything on the way out.
#
# Worth its quarter of a GPU-hour before committing eighteen of them. A run that
# comes back but has lost its optimizer state or replays batches it already saw
# is worse than one that does not come back, because it looks healthy.
#
#   sbatch scripts/sbatch/resume.sbatch
#   SIDE=alternate KILL_AFTER=900 sbatch scripts/sbatch/resume.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
SIDE="${SIDE:-bilateral}"
KILL_AFTER="${KILL_AFTER:-540}"     # past the first checkpoint at step 500
W="$DATA_p330/runs/resumecheck-$SLURM_JOB_ID"
RUN="apptainer exec --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF python -m harness.run"
ARGS="--anchor $SIDE --micro-batch 512 --out-dir $W \
      --data-path $DATA_p330/c4/c4_train.bin --val-path $DATA_p330/c4/c4_val.bin"

echo "node $(hostname)"
echo "### leg 1: the real anchor configuration, SIGKILLed after ${KILL_AFTER}s"
timeout -s KILL "$KILL_AFTER" $RUN $ARGS --max-steps 2000 > "$W.leg1" 2>&1
echo "    exit $? (137 = killed)"
DIR=$(ls -d "$W"/*/ 2>/dev/null | head -1)
apptainer exec --bind "$DATA_p330:$DATA_p330" "$SIF" python - "$DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + "log.jsonl")]
print(f"    reached step {max(r['step'] for r in rows if 'train_loss' in r)} before the kill")
ck = d + "checkpoint.pt"
print(f"    checkpoint: {os.path.getsize(ck)/1e9:.2f} GB" if os.path.exists(ck) else "    NO CHECKPOINT")
print(f"    stray .tmp left behind: {os.path.exists(ck + '.tmp')}")
PY

echo "### leg 2: the same command again -- it must resume, not restart"
timeout -s KILL 420 $RUN $ARGS --max-steps 2000 > "$W.leg2" 2>&1
apptainer exec --bind "$DATA_p330:$DATA_p330" "$SIF" python - "$DIR" <<'PY'
import json, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + "log.jsonl")]
resumes = [r["from_step"] for r in rows if r.get("event") == "resume"]
losses = [(r["step"], r["train_loss"]) for r in rows if "train_loss" in r]
print(f"    resume events: {resumes}")
print(f"    steps seen: {min(s for s, _ in losses)} .. {max(s for s, _ in losses)}")
print(f"    loss before the kill: {[round(l, 4) for s, l in losses if s < resumes[-1]][-3:]}")
print(f"    loss after resuming:  {[round(l, 4) for s, l in losses if s >= resumes[-1]][:3]}")
PY
rm -rf "$W" "$W".leg1 "$W".leg2
