#!/bin/bash
# Launch the paper's runs of the final form (v2.sbatch rows 216-256) and drop the held runs of the earlier form.
# Called once rows 214-215 (the final form at 3000 and 15000) pass Fast's 0.010 rule and keep the spectrum.
set -e
cd "${REPO:-$DATA_p330/Natural-Pion}"
S=scripts/paper/v2.sbatch; X="-x rtx6004"
scancel 403140 403141 403142 403143 403148 403146_188 403146_190 403146_191 403146_192 2>/dev/null || true
sbatch $X -t 01:30:00 --array=216-219,231,233-235,237-256%8 $S
sbatch $X -t 04:00:00 --array=220-223,232,236 $S
sbatch $X -t 02:30:00 --array=224-225 $S
j30=$(sbatch --parsable $X -t 06:00:00 --array=226-228 $S | cut -d_ -f1)
sbatch $X -t 10:00:00 --dependency=afterany:$j30 --array=229 $S
sbatch $X -t 16:00:00 --dependency=afterany:$j30 --array=230 $S
sbatch $X scripts/paper/timing.sbatch
echo "launched at $(date +%H:%M)"
