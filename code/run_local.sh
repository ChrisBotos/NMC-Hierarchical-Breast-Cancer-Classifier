#!/bin/bash
# Local runner for nested CV on a laptop (no SLURM required).
#
# Runs all (pipeline, repeat) jobs sequentially using the local config,
# then runs the analysis script to aggregate results and generate figures.
# Supports fold-level checkpointing: if a job is interrupted, re-running
# this script will resume each job from its last completed fold.
#
# Usage:
#     bash code/run_local.sh                              # default_run, local.yaml
#     bash code/run_local.sh my_experiment                # custom run name
#     bash code/run_local.sh my_experiment server.yaml    # custom run + config

set -euo pipefail

# Resolve project root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Arguments.
RUN_NAME="${1:-default_run}"
CONFIG_FILE="${2:-config_files/local.yaml}"

# Resolve config path relative to project root if not absolute.
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Robust conda activation with error handling.
# ---------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh || { echo "ERROR: conda.sh not found"; exit 1; }
conda activate tb_310 || { echo "ERROR: failed to activate tb_310"; exit 1; }

# Cap BLAS/OpenMP threads (laptop has limited cores).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Read pipeline names and repeat count from the config.
# ---------------------------------------------------------------------------
eval "$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
pipes = cfg['pipelines']['names']
print(f'N_REPEATS={cfg[\"cv\"][\"n_repeats\"]}')
print(f'PIPELINES=({\" \".join(repr(p) for p in pipes)})')
")"

N_PIPELINES=${#PIPELINES[@]}
TOTAL_JOBS=$(( N_PIPELINES * N_REPEATS ))

echo "========================================"
echo "Local nested CV run"
echo "  Run name:   $RUN_NAME"
echo "  Config:     $CONFIG_FILE"
echo "  Pipelines:  ${PIPELINES[*]}"
echo "  Repeats:    $N_REPEATS"
echo "  Total jobs: $TOTAL_JOBS"
echo "========================================"

# ---------------------------------------------------------------------------
# Run all (pipeline, repeat) jobs sequentially.
# ---------------------------------------------------------------------------
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

for PIPELINE in "${PIPELINES[@]}"; do
    for REPEAT in $(seq 1 "$N_REPEATS"); do
        COMPLETED=$((COMPLETED + 1))
        echo ""
        echo "--- Job $COMPLETED / $TOTAL_JOBS: $PIPELINE repeat $REPEAT ---"

        set +e
        python3 "$PROJECT_DIR/code/nested_cv_2x2_runner.py" \
            --pipeline "$PIPELINE" \
            --repeat "$REPEAT" \
            --config "$CONFIG_FILE" \
            --name "$RUN_NAME"
        EXIT_CODE=$?
        set -e

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "WARNING: $PIPELINE repeat $REPEAT failed (exit $EXIT_CODE)."
            FAILED=$((FAILED + 1))
        fi
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "========================================"
echo "All CV jobs complete."
echo "  Succeeded: $(( TOTAL_JOBS - FAILED )) / $TOTAL_JOBS"
if [[ $FAILED -gt 0 ]]; then
    echo "  Failed:    $FAILED"
fi
echo "  Elapsed:   ${HOURS}h ${MINS}m ${SECS}s"
echo "========================================"

# ---------------------------------------------------------------------------
# Run analysis to aggregate results and generate figures.
# ---------------------------------------------------------------------------
if [[ $FAILED -eq 0 ]]; then
    echo ""
    echo "Running analysis..."
    python3 "$PROJECT_DIR/code/analyse_nested_cv.py" \
        --name "$RUN_NAME" \
        --config "$CONFIG_FILE"
    echo "Analysis complete."
else
    echo ""
    echo "Skipping analysis ($FAILED jobs failed). Fix failures and re-run."
    exit 1
fi
