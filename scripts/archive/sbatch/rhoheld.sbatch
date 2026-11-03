#!/bin/bash
#SBATCH -A p330
#SBATCH -J rhoheld
#SBATCH --array=0-2%3
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is `rho` blind by construction, and what does the trust region really cost?
#
# TWO QUESTIONS, AND THE SECOND IS A REPAIR.
#
# 1. `rho` is evaluated on the batch whose gradient produced the step, so it
#    asks whether the quadratic model described *that* batch. A step several
#    times too long still describes it -- it is descending that batch's own
#    quadratic -- and the step here is 96.4% sampling noise, so the batch's
#    quadratic and the expected one are different objects. That is the obvious
#    reason a controller targeting `rho ~ 1` settles two to four times above
#    the swept optimum: it is steering by a quantity that cannot see the error
#    making the step too long.
#
#    HYPOTHESIS  the same-batch `rho` reads ~1 across a range of `eta` over
#                which the loss differs materially; a held-out `rho` does not.
#    PREDICTION  arms A and B agree on `rho_med` to within a few tenths, while
#                `rho_held_med` is materially lower at 0.077 than at 0.02.
#    CONTROL     A and B differ in `eta` and in nothing else -- same seed, same
#                node, same flags, controller off in both. If `rho_held` fails
#                to separate them, the idea dies here and no knob was added.
#
# 2. The 0.20 deficit charged to the trust region on 2026-08-28 is not sound.
#    The measurement forward fed `A` -- a forward pre-hook fires under
#    `no_grad` -- and `CovarianceAccumulator.observe` blends per *call*, not
#    per sample, so four `rho_micro = 128` chunks every five steps carried 44%
#    of `A`'s EMA weight from quarter-sized batches at post-step weights. The
#    controller arms ran at 44%, the fixed-rate control at 1%: two differences,
#    one conclusion. Fixed in `harness.train._recorders_off` and pinned by
#    `test_diagnostic_forward_leaves_the_covariance_alone`.
#
#    Arm C is arm A's own settings plus the controller, so the deficit is now a
#    one-variable comparison. Nothing else on disk can be reused for it: every
#    earlier fixed-rate run had `rho_every` unset.
#
# All three carry `rho_holdout`, which costs two extra chunked forwards per
# measurement -- about 6% of a step at `rho_every = 5` -- and, now that the
# recorders are silenced, moves no statistic.
#
# `log_every = 97` is coprime with `t_fac = 25`. At 100 every logged row sat on
# a refactorisation boundary, where the basis is fresh, `alpha = 1` and `rho`
# is at its lowest; every `rho` quoted before 2026-08-28 was that one phase.
#
# Same seed on all three: concurrent runs on one node that disagree read
# different corpus windows and starve each other for IOPS.
#
#   sbatch scripts/sbatch/rhoheld.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/rhoheld"

# arm      eta      controller
ETAS=(     2e-2     7.7e-2   2e-2  )
TRUST=(    false    false    true  )
LABEL=(   "A fixed at the swept optimum" \
          "B fixed where the controller lands" \
          "C the controller, against A as its clean control" )

ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"
TL="${TRUST[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}  ${LABEL[$SLURM_ARRAY_TASK_ID]}"
echo "eta=$ETA trust_lr=$TL rho_every=5 rho_micro=128 rho_holdout=true log_every=97"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" \
        --ngd-trust-lr "$TL" --ngd-trust-lr-lo 0.8 --ngd-trust-lr-hi 1.2 \
        --rho-every 5 --rho-micro 128 --rho-holdout true \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 97 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
