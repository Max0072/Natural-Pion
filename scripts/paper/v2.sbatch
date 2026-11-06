#!/bin/bash
#SBATCH -A p330
#SBATCH -J npv2
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Every run of the paper's second version (user, 2026-11-05): the query projection rotated per head (always), Natural
# Pion Fast in its final form unless a row varies something. One row per run:
#   TAG | OPTIMIZER | STEPS | ANGLE | SEED | BETA1 | RETRACTION | STATS | NP_KWARGS | EXTRA
# An empty RETRACTION / STATS is the optimizer class's own default (Fast: ns_delta, bf16; natural-pion = Algorithm 1:
# cayley, fp32). Runs land in $DATA_p330/runs/npv2/<TAG>/; scripts/paper/collect_npv2.sh copies the logs to
# runs-record/npv2 and refreshes the paper. Rows 0-98 were first submitted from v2_rest.sbatch (output under
# runs/npv2x), rows 99-110 from v2_perhead.sbatch (runs/npv2/s<steps>); collect_npv2.sh merges both.
#   sbatch -x rtx6004 -t <limit> --array=<rows>%4 scripts/paper/v2.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
F=natural-pion-fast; A=natural-pion
ROWS=(
# 0-5  retraction (3.4, 4.8) and statistics precision (4.7); the ns_delta/bf16 reference is npv2/s3000 seed 0
"ret-cayley|$F|3000|0.2|0|0.9|cayley|||"
"ret-ns|$F|3000|0.2|0|0.9|ns|||"
"ret-trunc|$F|3000|0.2|0|0.9|trunc|||"
"ret-trunc-15k|$F|15000|0.05|0|0.9|trunc|||"
"stats-fp32|$F|3000|0.2|0|0.9||fp32||"
"form46|$F|3000|0.2|0|0.9|cayley|fp32|{\"batch_sigma\":false,\"share_inputs\":false}|"
# 6-13 regularisation (5.4)
"reg-eps1e-6|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6"
"reg-eps1e-5|$F|3000|0.2|0|0.9||||--ngd-eps 1e-5"
"reg-eps1e-3|$F|3000|0.2|0|0.9||||--ngd-eps 1e-3"
"reg-eps1e-2|$F|3000|0.2|0|0.9||||--ngd-eps 1e-2"
"reg-damp1e-4|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-4"
"reg-damp1e-3|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-3"
"reg-damp1e-2|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-2"
"reg-damp1e-1|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-1"
# 14-18 true vs empirical Fisher (3.1 remark): sampled targets at three angles, empirical at the two others
"fisher-mc-0.1|$F|3000|0.1|0|0.9||||--ngd-fisher-mc-every 1"
"fisher-mc-0.2|$F|3000|0.2|0|0.9||||--ngd-fisher-mc-every 1"
"fisher-mc-0.4|$F|3000|0.4|0|0.9||||--ngd-fisher-mc-every 1"
"fisher-emp-0.1|$F|3000|0.1|0|0.9||||"
"fisher-emp-0.4|$F|3000|0.4|0|0.9||||"
# 19-24 Algorithm 1 (4.1, 4.2): momentum grid, no-momentum grid
"alg1-mom-0.05|$A|3000|0.05|0|0.9||||"
"alg1-mom-0.1|$A|3000|0.1|0|0.9||||"
"alg1-mom-0.2|$A|3000|0.2|0|0.9||||"
"alg1-nomom-0.2|$A|3000|0.2|0|0||||"
"alg1-nomom-0.4|$A|3000|0.4|0|0||||"
"alg1-nomom-0.8|$A|3000|0.8|0|0||||"
# 25   Algorithm 1 at 15 000 steps
"alg1-mom-15k|$A|15000|0.1|0|0.9||||"
# 26-31 amortised refresh (4.2): T_fac 25, without and with momentum
"tf25-nomom-0.4|$A|3000|0.4|0|0|||{\"t_fac\":25}|"
"tf25-nomom-0.8|$A|3000|0.8|0|0|||{\"t_fac\":25}|"
"tf25-nomom-1.6|$A|3000|1.6|0|0|||{\"t_fac\":25}|"
"tf25-mom-0.1|$A|3000|0.1|0|0.9|||{\"t_fac\":25}|"
"tf25-mom-0.2|$A|3000|0.2|0|0.9|||{\"t_fac\":25}|"
"tf25-mom-0.4|$A|3000|0.4|0|0.9|||{\"t_fac\":25}|"
# 32-35 T_fac 10, seeds 0-3
"tf10-s0|$A|3000|0.2|0|0.9|||{\"t_fac\":10}|"
"tf10-s1|$A|3000|0.2|1|0.9|||{\"t_fac\":10}|"
"tf10-s2|$A|3000|0.2|2|0.9|||{\"t_fac\":10}|"
"tf10-s3|$A|3000|0.2|3|0.9|||{\"t_fac\":10}|"
# 36-38 statistics windows (4.3)
"win-adlast6|$A|3000|0.2|0|0.9|||{\"t_fac\":25,\"stats_window\":6,\"beta_a\":0.5}|"
"win-betad09|$A|3000|0.2|0|0.9|||{\"t_fac\":25,\"beta_d\":0.9}|"
"win-betad095|$A|3000|0.2|0|0.9|||{\"t_fac\":25,\"beta_d\":0.95}|"
# 39-49 direction (5.4): X = F^-1 G against X = G, without and with momentum; angle grid (0.2 with momentum = npv2)
"dir-nat-nomom-0.1|$F|3000|0.1|0|0||||"
"dir-nat-nomom-0.2|$F|3000|0.2|0|0||||"
"dir-nat-nomom-0.4|$F|3000|0.4|0|0||||"
"dir-nat-mom-0.1|$F|3000|0.1|0|0.9||||"
"dir-nat-mom-0.4|$F|3000|0.4|0|0.9||||"
"dir-raw-nomom-0.1|$F|3000|0.1|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-nomom-0.2|$F|3000|0.2|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-nomom-0.4|$F|3000|0.4|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-mom-0.1|$F|3000|0.1|0|0.9|||{\"direction\":\"raw\"}|"
"dir-raw-mom-0.2|$F|3000|0.2|0|0.9|||{\"direction\":\"raw\"}|"
"dir-raw-mom-0.4|$F|3000|0.4|0|0.9|||{\"direction\":\"raw\"}|"
# 50-74 length rules (5.4): five-point angle grid each, x2 apart, centred on the old optimum
"len-before-3e-6|$F|3000|3e-6|0|0.9|||{\"length\":\"before\"}|"
"len-before-1e-5|$F|3000|1e-5|0|0.9|||{\"length\":\"before\"}|"
"len-before-3e-5|$F|3000|3e-5|0|0.9|||{\"length\":\"before\"}|"
"len-before-1e-4|$F|3000|1e-4|0|0.9|||{\"length\":\"before\"}|"
"len-before-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"before\"}|"
"len-rms-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-9.6e-3|$F|3000|9.6e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-fisher-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-9.6e-3|$F|3000|9.6e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-1.9e-2|$F|3000|1.9e-2|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-3.8e-2|$F|3000|3.8e-2|0|0.9|||{\"length\":\"fisher\"}|"
"len-quad-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-8e-3|$F|3000|8e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-1.6e-2|$F|3000|1.6e-2|0|0.9|||{\"length\":\"quad\"}|"
"len-fixed-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-8e-3|$F|3000|8e-3|0|0.9|||{\"length\":\"fixed\"}|"
# 75-80 each side by its own sigma; attention angle multiplier
"sides-each-0.1|$F|3000|0.1|0|0.9|||{\"sides\":\"each\"}|"
"sides-each-0.2|$F|3000|0.2|0|0.9|||{\"sides\":\"each\"}|"
"attn-x0.25|$F|3000|0.2|0|0.9|||{\"attn_mult\":0.25}|"
"attn-x0.5|$F|3000|0.2|0|0.9|||{\"attn_mult\":0.5}|"
"attn-x2|$F|3000|0.2|0|0.9|||{\"attn_mult\":2}|"
"attn-x4|$F|3000|0.2|0|0.9|||{\"attn_mult\":4}|"
# 81-91 schedule of the angle (AdamW keeps its cosine): best of three angles; warmup-stable-decay
"sched-constant-0.05|$F|3000|0.05|0|0.9||||--schedule constant"
"sched-constant-0.1|$F|3000|0.1|0|0.9||||--schedule constant"
"sched-constant-0.2|$F|3000|0.2|0|0.9||||--schedule constant"
"sched-linear-0.1|$F|3000|0.1|0|0.9||||--schedule linear"
"sched-linear-0.2|$F|3000|0.2|0|0.9||||--schedule linear"
"sched-linear-0.4|$F|3000|0.4|0|0.9||||--schedule linear"
"sched-warmcos-0.1|$F|3000|0.1|0|0.9||||--schedule warmcos"
"sched-warmcos-0.2|$F|3000|0.2|0|0.9||||--schedule warmcos"
"sched-warmcos-0.4|$F|3000|0.4|0|0.9||||--schedule warmcos"
"sched-wsd-0.2|$F|3000|0.2|0|0.9||||--schedule wsd --wsd-decay 0.2"
"sched-wsd-0.1|$F|3000|0.2|0|0.9||||--schedule wsd --wsd-decay 0.1"
# 92-93 every plane of X turned by the same angle (polar step)
"planes-equal-0.1|$F|3000|0.1|0|0.9|||{\"planes\":\"equal\"}|"
"planes-equal-0.2|$F|3000|0.2|0|0.9|||{\"planes\":\"equal\"}|"
# 94-96 statistics windows (4.3) on the T_fac = 25 form of the chain (statistics every step, EMA 0.95 / 0.5)
"win-dlast6|$F|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"d_window\":6}|"
"win-alast6|$F|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"a_window\":6}|"
"win-aevery2|$F|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.9025,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"a_every\":2}|"
# 97-98 profile of the optimizer step by part (Table 1): Algorithm 1 and its T_fac = 25 form, 400 steps
"prof-alg1|$A|3000|0.2|0|0.9||||--max-steps 400 --np-profile true"
"prof-tf25|$A|3000|0.2|0|0.9|||{\"t_fac\":25}|--max-steps 400 --np-profile true"
# 99-110 Section 5: 3000 and 15000 steps over seeds 0-3, the budget curve, full length
"s3000|$F|3000|0.2|0|0.9|||"
"s3000|$F|3000|0.2|1|0.9|||"
"s3000|$F|3000|0.2|2|0.9|||"
"s3000|$F|3000|0.2|3|0.9|||"
"s15000|$F|15000|0.05|0|0.9|||"
"s15000|$F|15000|0.05|1|0.9|||"
"s15000|$F|15000|0.05|2|0.9|||"
"s15000|$F|15000|0.05|3|0.9|||"
"s9000|$F|9000|0.05|0|0.9|||"
"s30000|$F|30000|0.05|0|0.9|||"
"s45000|$F|45000|0.05|0|0.9|||"
"s73242|$F|73242|0.05|0|0.9|||"
# 111 statistics precision at 15000 steps (4.7)
"stats-fp32-15k|$F|15000|0.05|0|0.9||fp32||"
# 112-123 the initial spectrum (Appendix D): power-law exponent alpha, flat, orthogonal; alpha 3 over seeds, 15000, full length
"init-pl2|$F|3000|0.2|0|0.9||||--init-pl-alpha 2"
"init-pl2.5|$F|3000|0.2|0|0.9||||--init-pl-alpha 2.5"
"init-pl3|$F|3000|0.2|0|0.9||||--init-pl-alpha 3"
"init-pl4|$F|3000|0.2|0|0.9||||--init-pl-alpha 4"
"init-pl5|$F|3000|0.2|0|0.9||||--init-pl-alpha 5"
"init-flat|$F|3000|0.2|0|0.9||||--init-pl-alpha inf"
"init-orth|$F|3000|0.2|0|0.9||||--init orthogonal"
"init-pl3|$F|3000|0.2|1|0.9||||--init-pl-alpha 3"
"init-pl3|$F|3000|0.2|2|0.9||||--init-pl-alpha 3"
"init-pl3|$F|3000|0.2|3|0.9||||--init-pl-alpha 3"
"init-pl3-15k|$F|15000|0.05|0|0.9||||--init-pl-alpha 3"
"init-pl3-73242|$F|73242|0.05|0|0.9||||--init-pl-alpha 3"
# 124-129 second wave: seeds 1-3 at the best angle of a grid (submit with --dependency=afterany:<grid job>)
"alg1-mom-seeds|$A|3000|best:alg1-mom|1|0.9||||"
"alg1-mom-seeds|$A|3000|best:alg1-mom|2|0.9||||"
"alg1-mom-seeds|$A|3000|best:alg1-mom|3|0.9||||"
"len-rms-seeds|$F|3000|best:len-rms|1|0.9|||{\"length\":\"rms\"}|"
"len-rms-seeds|$F|3000|best:len-rms|2|0.9|||{\"length\":\"rms\"}|"
"len-rms-seeds|$F|3000|best:len-rms|3|0.9|||{\"length\":\"rms\"}|"
# 130-134 grid edges found on 2026-11-05: angle 0.8 where 0.4 was best; smaller angles where X = G diverged
"dir-nat-nomom-0.8|$F|3000|0.8|0|0||||"
"dir-nat-mom-0.8|$F|3000|0.8|0|0.9||||"
"tf25-mom-0.8|$A|3000|0.8|0|0.9|||{\"t_fac\":25}|"
"dir-raw-nomom-0.05|$F|3000|0.05|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-nomom-0.025|$F|3000|0.025|0|0|||{\"direction\":\"raw\"}|"
# 135-136 TF32 (scripts/probes/tf32_check.py): everything but the Cayley step, and everything
"tf32-upstream|$F|3000|0.2|0|0.9|||{\"tf32\":\"upstream\"}|"
"tf32-all|$F|3000|0.2|0|0.9|||{\"tf32\":\"all\"}|"
# 137 TF32 before the Cayley step at 15000 steps: the second point of Fast's acceptance rule (within 0.010)
"tf32-upstream-15k|$F|15000|0.05|0|0.9|||{\"tf32\":\"upstream\"}|"
# 138-140 the peak angle at 15000 steps in this setting (0.05 is row s15000)
"theta15k-0.025|$F|15000|0.025|0|0.9||||"
"theta15k-0.1|$F|15000|0.1|0|0.9||||"
"theta15k-0.2|$F|15000|0.2|0|0.9||||"
# 141-142 F^-1 products in bf16 (Fast's acceptance rule: within 0.010 at 3000 and 15000)
"fisherbf16|$F|3000|0.2|0|0.9|||{\"fisher_dtype\":\"bf16\"}|"
"fisherbf16-15k|$F|15000|0.05|0|0.9|||{\"fisher_dtype\":\"bf16\"}|"
# 143-148 the RMS length rule: the 3000-step grid extended below its edge, and a 15000-step grid
"len-rms-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-1.5e-4|$F|3000|1.5e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-1.5e-4|$F|15000|1.5e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-3e-4|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-6e-4|$F|15000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-1.2e-3|$F|15000|1.2e-3|0|0.9|||{\"length\":\"rms\"}|"
# 149-150 which matrices want a larger angle (rms_angles.py: RMS gives v x2.3, o x1.7, q and k ~1): the angle rule
# with v and o at 2x; control, q and k at 2x
"vo-x2|$F|3000|0.2|0|0.9||||--np-angle-mult {\"wv\":2,\"wo\":2}"
"qk-x2|$F|3000|0.2|0|0.9||||--np-angle-mult {\"wq\":2,\"wk\":2}"
)
IFS='|' read -r TAG OPT STEPS LR SEED B1 RET STATS KW EXTRA <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
case "$RET" in ""|cayley|ns|ns_delta|trunc) ;; *) echo "row $SLURM_ARRAY_TASK_ID: bad RETRACTION '$RET'"; exit 1 ;; esac
case "$STATS" in ""|fp32|bf16) ;; *) echo "row $SLURM_ARRAY_TASK_ID: bad STATS '$STATS'"; exit 1 ;; esac
case "$KW" in ""|"{"*"}") ;; *) echo "row $SLURM_ARRAY_TASK_ID: bad NP_KWARGS '$KW'"; exit 1 ;; esac
# ANGLE "best:<prefix>": the angle of the best finished run among the tags <prefix>-<angle> (seed 0), read from their
# logs at start; refuses if any of them is unfinished. Submit such rows with --dependency=afterany:<the grid's job>.
if [ "${LR#best:}" != "$LR" ]; then
    LR=$(python3 - "${LR#best:}" <<'PY'
import glob, json, sys
prefix, best, todo = sys.argv[1], None, []
for m in glob.glob(f"/onyx/data/p330/runs/npv2*/{prefix}-*/*/manifest.json"):
    try:                                     # only the grid itself: <prefix>-<angle>, not e.g. <prefix>-15k
        float(m.split("/")[-3][len(prefix) + 1:])
    except ValueError:
        continue
    c = json.load(open(m))["config"]
    if c["seed"] != 0:
        continue
    vals = [json.loads(l) for l in open(m.replace("manifest.json", "log.jsonl")) if '"val_loss"' in l]
    if not vals or vals[-1]["step"] < c["train_steps"] - 1:
        todo.append(m.split("/")[-3]); continue
    if best is None or vals[-1]["val_loss"] < best[0]:
        best = (vals[-1]["val_loss"], c["lr"])
if todo or best is None:
    sys.exit(f"grid {prefix}: unfinished {todo or 'all'}")
print(best[1])
PY
) || { echo "cannot resolve the angle of $TAG"; exit 1; }
fi
EV=500; [ "$STEPS" -gt 3000 ] && EV=1000; [ "$STEPS" -gt 15000 ] && EV=2000
OPT_ARGS=()
[ -n "$RET" ] && OPT_ARGS+=(--np-retraction "$RET")
[ -n "$STATS" ] && OPT_ARGS+=(--np-stats-dtype "$STATS")
[ -n "$KW" ] && OPT_ARGS+=(--np-kwargs "$KW")
OUT="$DATA_p330/runs/npv2/$TAG"; mkdir -p "$OUT"
echo "row $SLURM_ARRAY_TASK_ID: $TAG $OPT steps $STEPS angle $LR seed $SEED beta1 $B1 $RET $STATS kw=$KW $EXTRA"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer "$OPT" "${OPT_ARGS[@]}" $EXTRA \
      --lr "$LR" --ngd-beta1 "$B1" --train-steps "$STEPS" --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
      --precision bf16 --seed "$SEED" --device cuda --log-every 23 --eval-every "$EV" --adamw-lr 8e-3 \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
