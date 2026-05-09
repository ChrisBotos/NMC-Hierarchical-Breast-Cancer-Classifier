#!/bin/bash
# Self-submitting SLURM wrapper for plateau ensemble size comparison.
#
# Runs en_nmc_pens with 3 different MAX_PLATEAU_SIZE values (50, 200, all)
# to compare Stage 2 balanced accuracy as a function of ensemble size.
# All variants run in the existing server_run_v2 directory to access
# pre-computed en_nmc base inner CV files. Same seeds (1001-1200) as the
# production run enable paired comparison.
#
# Three independent array jobs are submitted (one per plateau size).
# Each array has 200 tasks (one per repeat seed).
#
# Usage:
#     bash code/submit_pens_comparison.sh
#     bash code/submit_pens_comparison.sh server_run_v2
#     bash code/submit_pens_comparison.sh server_run_v2 configs/pens_comparison.yaml

set -euo pipefail

# Resolve project root relative to this script.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

# ---------------------------------------------------------------------------
# Robust conda activation with error handling.
# ---------------------------------------------------------------------------
activate_conda() {
    local prefix="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"
    local env="${CONDA_ENV_NAME:-tb_310}"
    source "$prefix/etc/profile.d/conda.sh" || { echo "ERROR: conda.sh not found at $prefix"; exit 1; }
    conda activate "$env" || { echo "ERROR: failed to activate $env"; exit 1; }
}

# ===========================================================================
# Inside SLURM: run one (repeat, plateau_size) job.
# MAX_PLATEAU_SIZE, RUN_NAME, CONFIG_FILE, SEED_START are exported env vars.
# ===========================================================================
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then

    # Trap signals for clean status output.
    handle_signal() {
        echo "[PENS-CMP-KILLED] array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} psize=${MAX_PLATEAU_SIZE} repeat=${REPEAT:-unknown} signal=$1"
        exit 143
    }
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap 'handle_signal HUP' HUP

    # Activate conda.
    activate_conda

    # Cap threads to prevent thrashing.
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    # Map array task ID directly to repeat seed.
    REPEAT=$(( SEED_START + SLURM_ARRAY_TASK_ID ))

    # Human-readable plateau size label.
    if [[ "$MAX_PLATEAU_SIZE" == "0" ]]; then
        PSIZE_LABEL="all"
    else
        PSIZE_LABEL="$MAX_PLATEAU_SIZE"
    fi

    JOB_START=$(date +%s)
    echo "========================================"
    echo "PENS Comparison - Job $((SLURM_ARRAY_TASK_ID + 1)) of ${N_REPEATS}"
    echo "Pipeline:       en_nmc_pens"
    echo "Plateau size:   ${PSIZE_LABEL} (--max-plateau-size ${MAX_PLATEAU_SIZE})"
    echo "Repeat seed:    ${REPEAT}"
    echo "Config:         ${CONFIG_FILE}"
    echo "Run:            ${RUN_NAME}"
    echo "Node:           $(hostname)"
    echo "Start:          $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # Build and run the Python command.
    PYTHON_CMD=(python3 "$PROJECT_DIR/code/hierarchical_nested_cv_runner.py" \
        --pipeline en_nmc_pens \
        --repeat "$REPEAT" \
        --config "$CONFIG_FILE" \
        --name "$RUN_NAME" \
        --max-plateau-size "$MAX_PLATEAU_SIZE" \
        --skip-if-complete)
    echo "Running: ${PYTHON_CMD[*]}"

    set +e
    "${PYTHON_CMD[@]}"
    EXIT_CODE=$?
    set -e

    JOB_END=$(date +%s)
    ELAPSED=$(( JOB_END - JOB_START ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    SECS=$(( ELAPSED % 60 ))

    echo "========================================"
    echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Elapsed:  ${HOURS}h ${MINS}m ${SECS}s"
    echo "Exit:     ${EXIT_CODE}"
    echo "========================================"

    echo "[PENS-CMP-DONE] array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} psize=${PSIZE_LABEL} repeat=${REPEAT} exit=${EXIT_CODE} elapsed=${ELAPSED}s"

    exit $EXIT_CODE
fi

# ===========================================================================
# Outside SLURM: read config, resolve run directory, and submit 3 array jobs.
# ===========================================================================

RUN_NAME="${1:-server_run_v2}"
CONFIG_FILE="${2:-configs/pens_comparison.yaml}"

# Resolve config path relative to project root if not absolute.
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
fi

# Read parameters from config.
eval "$(python3 -c "
import yaml, os
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
cv = cfg['cv']
sl = cfg['slurm']
env = cfg.get('environment', {})
conda_prefix = os.path.expanduser(env.get('conda_prefix', '~/miniconda3'))
conda_env = env.get('conda_env', 'tb_310')
print(f'N_REPEATS={cv[\"n_repeats\"]}')
print(f'SEED_START={cv.get(\"seed_start\", 1)}')
print(f'CONDA_PREFIX_DIR=\"{conda_prefix}\"')
print(f'CONDA_ENV_NAME=\"{conda_env}\"')
print(f'SLURM_MEM=\"{sl[\"mem\"]}\"')
print(f'SLURM_TIME=\"{sl[\"time\"]}\"')
print(f'SLURM_CPUS={sl[\"cpus_per_task\"]}')
print(f'SLURM_MAX_CONCURRENT={sl[\"max_concurrent_jobs\"]}')
print(f'SLURM_MAIL_USER=\"{sl.get(\"mail_user\", \"\")}\"')
")"

# Resolve the existing run directory.
RESULTS_DIR="$PROJECT_DIR/results"
RUN_DIR=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -1)

if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: Run directory for '$RUN_NAME' not found in $RESULTS_DIR"
    exit 1
fi

# Freeze config snapshot.
FROZEN_CONFIG="$RUN_DIR/config_snapshot_pens_comparison.yaml"
cp "$CONFIG_FILE" "$FROZEN_CONFIG"

PHASE_DIR="$RUN_DIR/hierarchical_nested_cv"
SLURM_LOG_DIR="$PHASE_DIR/logs/slurm"
mkdir -p "$SLURM_LOG_DIR"

# Plateau sizes to compare: 50, 200, 0 (all qualifying).
PLATEAU_SIZES=(50 200 0)

# Build mail arguments.
MAIL_ARGS=""
if [[ -n "$SLURM_MAIL_USER" ]]; then
    MAIL_ARGS="--mail-type=END,FAIL --mail-user=$SLURM_MAIL_USER"
fi

echo "========================================"
echo "Submitting PENS size comparison"
echo "  Run name:    $RUN_NAME"
echo "  Run dir:     $RUN_DIR"
echo "  Config:      $CONFIG_FILE"
echo "  Frozen as:   $FROZEN_CONFIG"
echo "  Seeds:       ${SEED_START}-$(( SEED_START + N_REPEATS - 1 ))"
echo "  Repeats:     $N_REPEATS"
echo "  Plateau sizes: ${PLATEAU_SIZES[*]}"
echo "  SLURM:       mem=${SLURM_MEM}  time=${SLURM_TIME}  cpus=${SLURM_CPUS}"
echo "  SLURM logs:  $SLURM_LOG_DIR"
echo "========================================"

# Save config snapshot to config.json.
TIMESTAMP=$(date -Iseconds)
SNAPSHOT_FILE="$RUN_DIR/config.json"

python3 -c "
import json, pathlib
config_path = pathlib.Path('$SNAPSHOT_FILE')
config = json.loads(config_path.read_text()) if config_path.exists() else {}
config['submit_pens_comparison'] = {
    'timestamp': '$TIMESTAMP',
    'run_name': '$RUN_NAME',
    'config_file': '$CONFIG_FILE',
    'frozen_config': '$FROZEN_CONFIG',
    'n_repeats': $N_REPEATS,
    'seed_start': $SEED_START,
    'seed_range': '$SEED_START-$(( SEED_START + N_REPEATS - 1 ))',
    'plateau_sizes': [$(IFS=,; echo "${PLATEAU_SIZES[*]}")],
    'pipeline': 'en_nmc_pens',
}
config_path.write_text(json.dumps(config, indent=2))
"

# Submit array jobs sequentially (chained with afterok dependencies)
# to stay within the 50-CPU limit.
PREV_JOB_ID=""
for PSIZE in "${PLATEAU_SIZES[@]}"; do
    if [[ "$PSIZE" == "0" ]]; then
        PSIZE_LABEL="all"
    else
        PSIZE_LABEL="$PSIZE"
    fi

    DEP_ARGS=""
    if [[ -n "$PREV_JOB_ID" ]]; then
        DEP_ARGS="--dependency=afterok:${PREV_JOB_ID}"
    fi

    PREV_JOB_ID=$(sbatch \
        --job-name="pens_p${PSIZE_LABEL}_${RUN_NAME}" \
        --array="0-$(( N_REPEATS - 1 ))%${SLURM_MAX_CONCURRENT}" \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS" \
        --mem="$SLURM_MEM" \
        --time="$SLURM_TIME" \
        --output="$SLURM_LOG_DIR/pens_p${PSIZE_LABEL}_%A_%a.out" \
        --error="$SLURM_LOG_DIR/pens_p${PSIZE_LABEL}_%A_%a.err" \
        $MAIL_ARGS \
        $DEP_ARGS \
        --export=NONE,PROJECT_DIR="$PROJECT_DIR",RUN_NAME="$RUN_NAME",CONFIG_FILE="$FROZEN_CONFIG",N_REPEATS="$N_REPEATS",SEED_START="$SEED_START",MAX_PLATEAU_SIZE="$PSIZE",CONDA_PREFIX_DIR="$CONDA_PREFIX_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",HOME="$HOME",USER="$USER",PATH="$PATH" \
        --parsable \
        "${BASH_SOURCE[0]}")

    if [[ -n "$DEP_ARGS" ]]; then
        echo "Submitted plateau_size=${PSIZE_LABEL}: job $PREV_JOB_ID (${N_REPEATS} tasks, chained after previous)"
    else
        echo "Submitted plateau_size=${PSIZE_LABEL}: job $PREV_JOB_ID (${N_REPEATS} tasks)"
    fi
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "After completion, compare BA2 across plateau sizes with:"
echo "  python3 -c \"import pandas as pd, glob; [print(f) for f in sorted(glob.glob('$RUN_DIR/hierarchical_nested_cv/data/fold_results_en_nmc_pens_p*_r*.csv'))[:5]]\""

exit 0
