#!/bin/bash
#SBATCH -A p330
#SBATCH -J cadence
#SBATCH --array=0-8%8
#SBATCH -w rtx6001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The basis is refreshed 12x slower than the statistics that build it.
#
# `alpha = quad/curv` is identically 1 on a fresh basis -- `X = F^-1 G` gives
# `curv = <X, F(X)> = <X, G> = quad` as an identity. Everything that lowers it
# is staleness. And the two numbers that set staleness disagree by an order of
# magnitude:
#
#     t_fac  = 25    the basis is rebuilt every 25 steps
#     beta_D = 0.5   D remembers two calls
#
# So the basis is built from a `D` that is entirely forgotten two steps later,
# and is then used for twenty-three more. Measured unaliased over 3000 steps at
# the good rate: `alpha = 1` on **9.4%** of layer-steps, median **0.34**. At
# too large a rate (eta = 0.077) the median is 0.026 -- the trust region is
# absorbing a fortyfold overshoot, which is what keeps that arm alive.
#
# EVERY `alpha` FIGURE THIS PROJECT HAS QUOTED WAS ALIASED, the same way `rho`
# was. `log_every = 100` is a multiple of `t_fac = 25`, so all 41048 diagnostic
# rows of the full-length run sit at `step % 25 == 1` -- one step after a
# refactorisation, the freshest the basis ever is. "alpha < 1 on 37.1% of
# layer-steps" is that one phase, not the run.
#
#     PREDICTION  median `alpha` rises towards 1 as `t_fac` approaches the
#                 horizon of the statistics, and as `beta_D` lengthens to meet
#                 `t_fac`. If it does, the method executes a real natural
#                 gradient step every step instead of one in twenty-five.
#     CONTROL     the corner `t_fac = 25, beta_D = 0.5` is the current
#                 algorithm, so the grid contains its own baseline.
#
# `log_every = 97` is coprime with 1, 5 and 25, so every cell is sampled across
# all phases of its own cycle. Choosing 100 again would re-create the bug this
# experiment exists to measure.
#
# WHY THE CONTROLLER IS ON. If `alpha` triples, the effective step `c = eta *
# alpha` triples at fixed `eta`, and the fast-refactorisation arms would lose on
# loss for taking too long a step rather than for anything about cadence --
# comparing configurations at a shared `eta` is the error this project has a
# standing rule against. Job 298962 measured the rule on `eta` converging and
# beating a swept fixed rate (3.9312 against 3.9465), so each cell is allowed to
# find its own. The primary readout is `alpha` regardless; loss and the settled
# `eta` are secondary.
#
# COST: `t_fac = 1` factorises every step. The factorisation is ~12% of a step
# amortised over 25, so those three arms are several times slower. That is why
# the walltime is six hours and why 5 is in the grid as the practical middle.
#
#   sbatch -p rtx scripts/sbatch/cadence.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/cadence"

TFAC=(  1     1     1     5     5     5    25    25    25   )
BETAD=( 0.5   0.9   0.95  0.5   0.9   0.95 0.5   0.9   0.95 )

T="${TFAC[$SLURM_ARRAY_TASK_ID]}"
B="${BETAD[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}  t_fac=$T  beta_D=$B  eta adaptive from 0.02"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr 2e-2 \
        --ngd-t-fac "$T" --ngd-beta-backward "$B" \
        --ngd-trust-lr true --ngd-trust-lr-lo 0.8 --ngd-trust-lr-hi 1.2 \
        --rho-every 5 --rho-micro 128 \
        --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 97 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
