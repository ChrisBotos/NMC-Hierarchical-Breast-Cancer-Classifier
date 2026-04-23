#!/bin/bash
# Self-submitting SLURM wrapper for hierarchical_nested_cv_2x2_runner.py.
#
# When run outside SLURM: resolves the run directory, reads job parameters
# from the YAML config file, freezes a config snapshot, creates SLURM log
# directories, and submits itself as an array job.
# When run inside SLURM: maps the array task ID to (pipeline, repeat)
# and runs the Python script against the frozen config snapshot.
#
# Stage 1 (HER2+ vs rest) is fixed: KW+RF, k=5, no tuning.
# Stage 2 (HR+ vs TN) uses the pipeline specified by the array task mapping.
# The goal is to find the best Stage 2 pipeline across 4 options x 50 repeats.
#
# Usage:
#     bash code/submit_hierarchical_nested_cv.sh                                       # default_run, config_files/server.yaml
#     bash code/submit_hierarchical_nested_cv.sh my_experiment                         # custom run name, server config
#     bash code/submit_hierarchical_nested_cv.sh my_experiment config_files/local.yaml  # custom run name + local config
#
# Array mapping:
#     Task ID = pipeline_index * n_repeats + (repeat - 1)
#     pipeline_index: 0=kw_nmc, 1=kw_rf, 2=en_nmc, 3=en_rf
#     --pipeline controls the Stage 2 pipeline (Stage 1 is always KW+RF k=5).

set -euo pipefail

# Resolve project root relative to this script (only meaningful outside SLURM).
# Inside SLURM, PROJECT_DIR arrives as an exported env var because BASH_SOURCE
# points to the spool copy, not the original project tree.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

# ---------------------------------------------------------------------------
# Robust conda activation with error handling.
# CONDA_PREFIX_DIR and CONDA_ENV_NAME are set from the YAML config.
# ---------------------------------------------------------------------------
activate_conda() {
    local prefix="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"
    local env="${CONDA_ENV_NAME:-tb_310}"
    source "$prefix/etc/profile.d/conda.sh" || { echo "ERROR: conda.sh not found at $prefix"; exit 1; }
    conda activate "$env" || { echo "ERROR: failed to activate $env"; exit 1; }
}

# ===========================================================================
# Inside SLURM: activate environment and run one (pipeline, repeat) job.
# RUN_NAME, CONFIG_FILE, and REPEATS_PER_PIPELINE are exported env vars.
# CONFIG_FILE points to the frozen snapshot (config_snapshot_hierarchical.yaml).
# ===========================================================================
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then

    # Trap signals to emit a clear status line before exit.
    # TERM: SLURM time limit / preemption. INT: Ctrl-C (interactive debugging). HUP: terminal close.
    handle_signal() {
        echo "[HIER-NCV-TASK-KILLED] array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stage2_pipeline=${PIPELINE:-unknown} repeat=${REPEAT:-unknown} signal=$1"
        exit 143
    }
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap 'handle_signal HUP' HUP

    # Activate conda with error handling.
    activate_conda

    # Cap BLAS/OpenMP threads to prevent thrashing with concurrent jobs.
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    # Read pipeline names from exported PIPELINES_STR (space-delimited, no per-task Python).
    read -ra PIPELINES <<< "$PIPELINES_STR"
    N_PIPELINES=${#PIPELINES[@]}
    TOTAL_JOBS=$(( N_PIPELINES * REPEATS_PER_PIPELINE ))

    # Map array task ID to (pipeline_index, repeat).
    PIPELINE_IDX=$(( SLURM_ARRAY_TASK_ID / REPEATS_PER_PIPELINE ))
    REPEAT=$(( SLURM_ARRAY_TASK_ID % REPEATS_PER_PIPELINE + 1 ))
    PIPELINE="${PIPELINES[$PIPELINE_IDX]}"

    JOB_START=$(date +%s)
    echo "========================================"
    echo "Hierarchical CV - Job $((SLURM_ARRAY_TASK_ID + 1)) of ${TOTAL_JOBS}"
    echo "Stage 1:  kw_rf k=5 (fixed)"
    echo "Stage 2:  ${PIPELINE}"
    echo "Repeat:   ${REPEAT} / ${REPEATS_PER_PIPELINE}"
    echo "Config:   ${CONFIG_FILE}"
    echo "Run:      ${RUN_NAME}"
    echo "Node:     $(hostname)"
    echo "CPUs:     ${SLURM_CPUS_PER_TASK:-1}"
    echo "Memory:   ${SLURM_MEM_PER_NODE:-unknown}MB"
    echo "Work dir: $(pwd)"
    echo "Start:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # Build and echo the exact command for debugging.
    PYTHON_CMD=(python3 "$PROJECT_DIR/code/hierarchical_nested_cv_2x2_runner.py" \
        --pipeline "$PIPELINE" \
        --repeat "$REPEAT" \
        --config "$CONFIG_FILE" \
        --name "$RUN_NAME")
    echo "Running: ${PYTHON_CMD[*]}"

    # Run with explicit exit-code capture (set +e so the finish block always runs).
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

    # Greppable machine-parseable completion marker.
    echo "[HIER-NCV-TASK-DONE] array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stage2_pipeline=${PIPELINE} repeat=${REPEAT} exit=${EXIT_CODE} elapsed=${ELAPSED}s"

    exit $EXIT_CODE
fi

# ===========================================================================
# Outside SLURM: read config, set up run directory, and submit.
# ===========================================================================

# Arguments (positional, only used outside SLURM).
RUN_NAME="${1:-default_run}"
CONFIG_FILE="${2:-config_files/server.yaml}"

# Resolve config path relative to project root if not absolute.
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Read all job parameters from YAML config in a single Python invocation.
# ---------------------------------------------------------------------------
eval "$(python3 -c "
import yaml, os
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
cv = cfg['cv']
sl = cfg['slurm']
env = cfg.get('environment', {})
pipes = cfg['pipelines']['names']
conda_prefix = os.path.expanduser(env.get('conda_prefix', '~/miniconda3'))
conda_env = env.get('conda_env', 'tb_310')
print(f'REPEATS_PER_PIPELINE={cv[\"n_repeats\"]}')
print(f'CONDA_PREFIX_DIR=\"{conda_prefix}\"')
print(f'CONDA_ENV_NAME=\"{conda_env}\"')
print(f'SLURM_MEM=\"{sl[\"mem\"]}\"')
print(f'SLURM_TIME=\"{sl[\"time\"]}\"')
print(f'SLURM_CPUS={sl[\"cpus_per_task\"]}')
print(f'SLURM_MAX_CONCURRENT={sl[\"max_concurrent_jobs\"]}')
print(f'SLURM_MAIL_USER=\"{sl.get(\"mail_user\", \"\")}\"')
print(f'PIPELINES=({\" \".join(repr(p) for p in pipes)})')
")"

N_PIPELINES=${#PIPELINES[@]}
TOTAL_JOBS=$(( N_PIPELINES * REPEATS_PER_PIPELINE ))

# ---------------------------------------------------------------------------
# Resolve or create the run directory (bash equivalent of get_run_dirs).
# ---------------------------------------------------------------------------
resolve_run_dir() {
    local results_dir="$PROJECT_DIR/results"
    mkdir -p "$results_dir"

    # Search for existing *_<run_name> directory.
    local match
    match=$(find "$results_dir" -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -1)

    if [[ -n "$match" ]]; then
        echo "$match"
    else
        local today
        today=$(date +%Y-%m-%d)
        local new_dir="$results_dir/${today}_${RUN_NAME}"
        mkdir -p "$new_dir"
        echo "$new_dir"
    fi
}

# Validate that the runner script exists before submitting.
RUNNER_SCRIPT="$PROJECT_DIR/code/hierarchical_nested_cv_2x2_runner.py"
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
    echo "ERROR: runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

RUN_DIR=$(resolve_run_dir)
PHASE_DIR="$RUN_DIR/hierarchical_nested_cv_2x2"
SLURM_LOG_DIR="$PHASE_DIR/logs/slurm"
mkdir -p "$SLURM_LOG_DIR"

# Freeze the YAML config so all array tasks read a consistent snapshot.
FROZEN_CONFIG="$RUN_DIR/config_snapshot_hierarchical.yaml"
cp "$CONFIG_FILE" "$FROZEN_CONFIG"

# Save config snapshot to config.json.
SNAPSHOT_FILE="$RUN_DIR/config.json"
TIMESTAMP=$(date -Iseconds)

python3 -c "
import json, pathlib
config_path = pathlib.Path('$SNAPSHOT_FILE')
config = json.loads(config_path.read_text()) if config_path.exists() else {}
config['submit_hierarchical_nested_cv'] = {
    'timestamp': '$TIMESTAMP',
    'run_name': '$RUN_NAME',
    'config_file': '$CONFIG_FILE',
    'frozen_config': '$FROZEN_CONFIG',
    'repeats_per_pipeline': $REPEATS_PER_PIPELINE,
    'stage1': 'kw_rf k=5 (fixed)',
    'stage2_pipelines': $(printf '%s\n' "${PIPELINES[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))"),
    'total_jobs': $TOTAL_JOBS,
    'slurm': {
        'mem': '$SLURM_MEM',
        'time': '$SLURM_TIME',
        'cpus_per_task': $SLURM_CPUS,
        'max_concurrent_jobs': $SLURM_MAX_CONCURRENT,
    },
}
config_path.write_text(json.dumps(config, indent=2))
"

echo "========================================"
echo "Submitting hierarchical nested CV array job"
echo "  Run name:    $RUN_NAME"
echo "  Run dir:     $RUN_DIR"
echo "  Config file: $CONFIG_FILE"
echo "  Frozen as:   $FROZEN_CONFIG"
echo "  Stage 1:     kw_rf k=5 (fixed, no tuning)"
echo "  Stage 2:     ${PIPELINES[*]} (4 pipelines, tuned via inner CV)"
echo "  Jobs:        $TOTAL_JOBS (${N_PIPELINES} stage2 pipelines x ${REPEATS_PER_PIPELINE} repeats)"
echo "  SLURM:       mem=${SLURM_MEM}  time=${SLURM_TIME}  cpus=${SLURM_CPUS}  max_concurrent=${SLURM_MAX_CONCURRENT}"
echo "  SLURM logs:  $SLURM_LOG_DIR"
echo "========================================"

# Build mail arguments only if a mail user is configured.
MAIL_ARGS=""
if [[ -n "$SLURM_MAIL_USER" ]]; then
    MAIL_ARGS="--mail-type=END,FAIL --mail-user=$SLURM_MAIL_USER"
fi

# Export frozen config path so array tasks read the snapshot, not the live YAML.
ARRAY_JOB_ID=$(sbatch \
    --job-name="hncv_${RUN_NAME}" \
    --array="0-$(( TOTAL_JOBS - 1 ))%${SLURM_MAX_CONCURRENT}" \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS" \
    --mem="$SLURM_MEM" \
    --time="$SLURM_TIME" \
    --output="$SLURM_LOG_DIR/hier_ncv_%A_%a.out" \
    --error="$SLURM_LOG_DIR/hier_ncv_%A_%a.err" \
    $MAIL_ARGS \
    --export=NONE,PROJECT_DIR="$PROJECT_DIR",RUN_NAME="$RUN_NAME",CONFIG_FILE="$FROZEN_CONFIG",REPEATS_PER_PIPELINE="$REPEATS_PER_PIPELINE",PIPELINES_STR="${PIPELINES[*]}",CONDA_PREFIX_DIR="$CONDA_PREFIX_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",HOME="$HOME",USER="$USER",PATH="$PATH" \
    --parsable \
    "${BASH_SOURCE[0]}")

echo "Array job submitted: $ARRAY_JOB_ID"
echo "Monitor with: squeue -u \$USER -j $ARRAY_JOB_ID"
echo "Run analysis after completion with:"
echo "  python3 code/analyse_nested_cv.py --name $RUN_NAME --config $FROZEN_CONFIG"

exit 0
