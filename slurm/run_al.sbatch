#!/bin/bash
# One job = the whole AL loop on one node; simulations run in parallel inside the allocation.
# The job resubmits itself until the experiment is finished:
#   * run_al.py exits 3 when the remaining walltime is too short for another round,
#   * Slurm sends USR1 10 min before the limit (--signal) if a round runs longer than expected.
# State lives in the workdir, so the next job resumes where this one stopped.
#
#   mkdir -p slurm_logs
#   sbatch slurm/run_al.sbatch --task pd --workdir runs/pd_gp --surrogate dirichlet_gp
#
#SBATCH --job-name=pnm_al
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm_logs/%x_%j.out
##SBATCH --partition=msismall
##SBATCH --account=<your_group>

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 1
mkdir -p slurm_logs

# Runtime libraries karst needs (MUMPS / BLAS ...), as in the collaborators' karst build:
# module load ...

# Python env created once by setup_msi.sh (nothing is installed inside the job).
if [ -z "${PYTHON:-}" ]; then
    ENV_DIR=${PNM_AL_ENV:-$PWD/env}
    if [ ! -x "$ENV_DIR/bin/python" ]; then
        echo "ERROR: no Python env at $ENV_DIR - run 'bash setup_msi.sh' on a login node first"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    PYTHON="$ENV_DIR/bin/python"
fi
export HF_HUB_OFFLINE=1          # compute nodes: TabICL reads the checkpoint prefetched by setup_msi.sh
SBATCH=${SBATCH:-sbatch}
SCRIPT=${PNM_AL_SBATCH_SCRIPT:-slurm/run_al.sbatch}
RESUBMITS=${PNM_AL_RESUBMITS:-0}
MAX_RESUBMITS=${PNM_AL_MAX_RESUBMITS:-30}

resubmit() {
    if [ "$RESUBMITS" -ge "$MAX_RESUBMITS" ]; then
        echo "not resubmitting: already resubmitted $RESUBMITS times (PNM_AL_MAX_RESUBMITS)"; return 1
    fi
    PNM_AL_RESUBMITS=$((RESUBMITS + 1)) "$SBATCH" --export=ALL "$SCRIPT" "$@"
}

ARGS=("$@")
on_usr1() {
    echo "walltime almost reached: stopping the loop and resubmitting"
    kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
    resubmit "${ARGS[@]}"
    exit 0
}
trap on_usr1 USR1

"$PYTHON" run_al.py run "$@" &
PID=$!
wait "$PID"
RC=$?

if [ "$RC" -eq 3 ]; then
    resubmit "$@" || exit 1
    exit 0
fi
exit "$RC"
