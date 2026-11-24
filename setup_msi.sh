#!/bin/bash
# One-time setup on an MSI login node: the Python environment for the AL framework.
#
#   bash setup_msi.sh
#
# karst itself is NOT built here (the collaborators have their own build procedure).
# Afterwards either put the binary at karst/bin/karst or pass --karst-cmd /path/to/karst.
#
# Options (environment variables):
#   PYTHON_MODULE    module to load first, e.g. "python3/3.11.x" (check `module avail python`)
#   PYTHON           interpreter used to create the env (default: python3), must be >= 3.11
#   ENV_DIR          where the virtual env goes (default: ./env)
#   TORCH_INDEX_URL  e.g. https://download.pytorch.org/whl/cpu to install the CPU-only torch
#                    (default pip wheel includes CUDA, ~2 GB)
#   SKIP_TABICL_PREFETCH=1   do not download the TabICL checkpoint now
#   SKIP_TESTS=1             do not run the test suite at the end
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(pwd)
ENV_DIR=${ENV_DIR:-$REPO/env}

if [ -n "${PYTHON_MODULE:-}" ]; then
    module load "$PYTHON_MODULE"
fi
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "ERROR: '$PY' not found; set PYTHON or PYTHON_MODULE"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
    echo "ERROR: $("$PY" --version 2>&1) is too old; requirements.txt (numpy 2.4) needs Python >= 3.11"; exit 1; }

# ----------------------------------------------------------- environment ----
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "creating virtual env in $ENV_DIR"
    "$PY" -m venv "$ENV_DIR"
fi
ENV_PY="$ENV_DIR/bin/python"
"$ENV_PY" -m pip install --upgrade pip
if [ -n "${TORCH_INDEX_URL:-}" ]; then
    TORCH_PIN=$(grep -E '^torch==' requirements.txt || true)
    [ -n "$TORCH_PIN" ] && "$ENV_PY" -m pip install "$TORCH_PIN" --index-url "$TORCH_INDEX_URL"
fi
"$ENV_PY" -m pip install -r requirements.txt
"$ENV_PY" -m pip install -e . --no-deps          # versions come from requirements.txt only

# ---------------------------------------------- TabICL checkpoint (online) ----
# Compute nodes are usually offline; download the checkpoint now into the HF cache.
if [ "${SKIP_TABICL_PREFETCH:-0}" != 1 ]; then
    "$ENV_PY" -m pnm_al.prefetch_tabicl || echo "WARNING: TabICL prefetch failed; --surrogate tabicl will not work offline"
fi

# ------------------------------------------------------------------ checks ----
if [ "${SKIP_TESTS:-0}" != 1 ]; then
    "$ENV_PY" -m unittest discover -s tests       # uses the fake karst, no real karst needed
fi
if [ -x "$REPO/karst/bin/karst" ]; then
    echo "karst binary found: $REPO/karst/bin/karst"
else
    echo "NOTE: no karst/bin/karst yet. Build karst with the collaborators' procedure, then copy or link"
    echo "      the binary to karst/bin/karst, or pass --karst-cmd /path/to/karst to run_al.py."
fi
echo
echo "done. Jobs use this env automatically (slurm/run_al.sbatch looks for $ENV_DIR);"
echo "set PNM_AL_ENV=/other/env if you put it elsewhere."
