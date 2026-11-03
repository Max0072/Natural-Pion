#!/bin/bash
#SBATCH -A p330
#SBATCH -J fisher
#SBATCH --array=0-7%8
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The true Fisher against the empirical one, each on its own `eta` grid.
#
# THE QUESTION IS THE USER'S. Natural gradient is supposed to converge much
# faster than ordinary methods and here it does not obviously do so, so perhaps
# the Fisher is wrong. Two candidates were ruled out first, by measurement
# rather than by argument:
#
#   * the spectral floor is not eating it. `floor_bites.py` on the finished
#     full-length checkpoint: at `eps = 1e-4` the floor touches 0.2% of `A` at
#     the median, the denominator keeps 3443x of dynamic range and `iso-cos` is
#     0.82. The preconditioner is alive. (A tail of layers is genuinely
#     degenerate -- 96.5% floored on one -- but that is a handful out of 56.)
#   * the K-FAC independence failure is real and measured at 30-170x, but a
#     global magnitude error is absorbed by `eta`; only its ~5x per-layer
#     spread is not.
#
# What is left untested is that every headline run uses the **empirical**
# Fisher. `mc_every = 0` in all of them. The true Fisher was evaluated exactly
# once, in `mcfisher2`, at 150 steps -- the protocol this project has caught
# inverting rankings three times -- and confounded with two other changes. The
# inadequacy of the empirical Fisher as a curvature matrix away from the
# optimum is the standard published reason natural gradient underdelivers.
#
# EACH ARM ON ITS OWN GRID, AND THE GRIDS ARE MEASURED, NOT GUESSED. Job 299571
# ran 100 steps of each at a shared `eta` and read the scale off `pred_drop`:
#
#     mc = 0    median pred_drop  0.0696
#     mc = 1    median pred_drop  0.5717      x8.2
#
# `quad = <G, F^-1 G>` scales as `1/k` when `F` scales as `k`, so the true
# Fisher is 8.2x *smaller* here, the natural step 8.2x longer, and the optimum
# `eta` about 8x lower. A grid centred on 0.02 would have missed entirely --
# which is the whole reason the calibration was run.
#
# THE mc=0 HALF IS ALSO THE CONTROL THE TRUST REGION NEEDS. Job 298962 showed
# the rule on `eta` beating a fixed 0.02 (3.9312 against 3.9465) and settling
# at 0.006, well below the rate an old sweep called optimal. Claiming it beats
# the *best* fixed rate needs a fixed grid on current code that includes 0.006.
# These four arms are that grid.
#
# Same seed on all eight: concurrent runs on one node that disagree read
# different corpus windows and starve each other for IOPS.
#
#   sbatch -p rtx scripts/sbatch/fisher.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/fisher"

#      ---------- empirical ----------   ------------ true (MC) ------------
MC=(   0        0        0        0        1         1         1        1     )
ETAS=( 2e-3     6e-3     2e-2     6e-2     8e-4      2.4e-3    7e-3     2e-2  )

M="${MC[$SLURM_ARRAY_TASK_ID]}"
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}  mc_every=$M  eta=$ETA  3000 steps"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" \
        --ngd-fisher-mc-every "$M" \
        --ngd-trust-lr false \
        --rho-every 5 --rho-micro 128 --rho-holdout true \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 97 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
