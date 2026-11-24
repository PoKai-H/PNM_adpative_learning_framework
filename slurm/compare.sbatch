#!/bin/bash
# Strategy comparison: each array task is an independent AL experiment with its own workdir;
# every other option is identical. Extra arguments are passed to all variants.
#
#   mkdir -p slurm_logs
#   sbatch --array=0-3 slurm/compare.sbatch --task pd --max-rounds 15
#
#SBATCH --job-name=pnm_al_cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm_logs/%x_%A_%a.out
##SBATCH --partition=msismall
##SBATCH --account=<your_group>

NAMES=(gp_diverse gp_topk tabicl_diverse tabicl_topk)
VARIANTS=(
    "--surrogate dirichlet_gp --batch-strategy diverse"
    "--surrogate dirichlet_gp --batch-strategy top_k"
    "--surrogate tabicl --batch-strategy diverse"
    "--surrogate tabicl --batch-strategy top_k"
)
i=${SLURM_ARRAY_TASK_ID:?run this as a job array, e.g. sbatch --array=0-3}
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
# A resumed variant is resubmitted as a plain run_al.sbatch job (same arguments, own workdir).
export PNM_AL_SBATCH_SCRIPT=slurm/run_al.sbatch
# shellcheck disable=SC2086
exec bash slurm/run_al.sbatch --workdir "runs/cmp_${NAMES[$i]}" ${VARIANTS[$i]} "$@"
