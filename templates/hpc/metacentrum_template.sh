#!/bin/bash
#PBS -N cmgan_training
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:scratch_local=20gb
#PBS -l walltime=03:30:00
#PBS -j oe
#PBS -m ae
#PBS -M varhavit@fel.cvut.cz

# -------------------------------------------------------------------------------------
# THIS FILE IS HARDCODED TEMPLATE OF HOW PBS SCRIPT FOR JOB SUBMITTING MAY LOOK LIKE
# -------------------------------------------------------------------------------------

# cluster=fer needed to run only on fer cluster, because of paths problem, need to be removed after fix
# gpu_mem=16gb:gpu_cap=compute_80 should be set to avoid old GPUs

set -euo pipefail

LOGROOT="$PBS_O_HOME/pbs_logs/cmgan"
mkdir -p "$LOGROOT"

PBSLOG="$LOGROOT/pbs_${PBS_JOBID}.log"
TRAINLOG="$LOGROOT/training_${PBS_JOBID}.log"

exec > >(tee -a "$PBSLOG") 2>&1

echo "Logging to: $PBSLOG"
echo "Training log: $TRAINLOG"
echo "JobID      : $PBS_JOBID"
echo "Host       : $(hostname -f)"

test -n "${SCRATCHDIR:-}" || { echo >&2 "SCRATCHDIR not set"; exit 1; }

ls /auto/projects-du-praha/CTU_Speech_Lab > /dev/null 2>&1 || true

# the problem with paths, some frontends have auto/ some /projects/ and some /projects-du-praha/
# consult metacentrum.cz

CANDIDATES=(
  "/storage/projects-du-praha/CTU_Speech_Lab/workspace/varhavit/speech-enhancement-cmgan"
  "/storage/projects/CTU_Speech_Lab/workspace/varhavit/speech-enhancement-cmgan"
  "/auto/projects-du-praha/CTU_Speech_Lab/workspace/varhavit/speech-enhancement-cmgan"
  "${PBS_O_WORKDIR:-}"
)

PROJECT_DIR=""
for d in "${CANDIDATES[@]}"; do
  if [ -n "$d" ] && [ -d "$d" ]; then
      PROJECT_DIR="$d"
      break
  fi
done

[ -n "$PROJECT_DIR" ] || { echo "No PROJECT_DIR found on this node."; exit 1; }

cd "$PROJECT_DIR"
echo "Using PROJECT_DIR: $PROJECT_DIR"

WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
BASE="$(dirname "$(dirname "$WORKSPACE_DIR")")"

PYTHON_EXEC="$WORKSPACE_DIR/envs/audio_env/bin/python"
test -x "$PYTHON_EXEC" || { echo >&2 "Python not found/executable: $PYTHON_EXEC"; exit 1; }

export SAVE_DIR="$SCRATCHDIR/checkpoints"
mkdir -p "$SAVE_DIR"

FINAL_CHECKPOINT_DIR="$PBS_O_HOME/cmgan_checkpoints/${PBS_JOBID}"
mkdir -p "$FINAL_CHECKPOINT_DIR"

rescue_checkpoints() {
    echo "=================================================="
    echo "Job ending or Walltime reached! Rescuing checkpoints..."

    cp -r "$SAVE_DIR/"* "$FINAL_CHECKPOINT_DIR/" || echo "WARNING: Copy failed or no files to copy!"

    echo "Checkpoints successfully copied to $FINAL_CHECKPOINT_DIR"
    echo "Cleaning local scratch..."
    clean_scratch
    echo "=================================================="
}

trap 'rescue_checkpoints' EXIT TERM

RUNTIME_CFG="$SCRATCHDIR/metacentrum.runtime.yaml"
sed \
  -e "s|^  root:.*|  root: \"$BASE\"|g" \
  -e "s|^  workspace:.*|  workspace: \"$WORKSPACE_DIR\"|g" \
  -e "s|^  scratch:.*|  scratch: \"$SCRATCHDIR\"|g" \
  -e "s|^  save_dir:.*|  save_dir: \"$SAVE_DIR\"|g" \
  config/metacentrum.yaml > "$RUNTIME_CFG"

echo "Runtime config written to: $RUNTIME_CFG"

export PYTHONPATH="$WORKSPACE_DIR:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting CMGAN Training..."

"$PYTHON_EXEC" -u -m enhancement.main --config "$RUNTIME_CFG" 2>&1 | tee -a "$TRAINLOG"