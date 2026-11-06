#!/bin/bash
#SBATCH -A p330
#SBATCH -J asyncomp
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/async_compile_check.py. TRITON_LIBCUDA_PATH points Inductor at the driver apptainer --nv mounts.
#   sbatch -x rtx6004 scripts/probes/async_compile_check.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    bash -c 'ls /.singularity.d/libs | grep -i libcuda; python scripts/probes/async_compile_check.py'
