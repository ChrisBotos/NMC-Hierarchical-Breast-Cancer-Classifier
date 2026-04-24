#!/bin/bash
# Self-submitting SLURM wrapper for hierarchical_nested_cv_runner.py.
#
# When run outside SLURM: resolves the run directory, reads job parameters
# from the YAML config file, freezes a config snapshot, creates SLURM log
# directories, and submits itself as an array job.
# When run inside SLURM: maps the array task ID to (pipeline, repeat)
# and runs the Python script against the frozen config snapshot.
#
# Stage 1 (HER2+ vs rest) is fixed: KW+RF, k=5, with threshold calibration.
# Stage 2 (HR+ vs TN) uses the pipeline specified by the array task mapping.
# Pipeline count is auto-derived from the config file (typically 8+ variants).
#
# Usage:
#     bash code/submit_hierarchical_nested_cv.sh                                            # default_run, server config, all pipelines
#     bash code/submit_hierarchical_nested_cv.sh my_experiment                              # custom run name
#     bash code/submit_hierarchical_nested_cv.sh my_experiment config_files/local.yaml      # custom run name + config
#     bash code/submit_hierarchical_nested_cv.sh --skip kw_rf,en_rf my_experiment           # skip specific pipelines
#     bash code/submit_hierarchical_nested_cv.sh --only kw_nmc_pens,en_nmc_pens my_run      # only submit listed pipelines
#     bash code/submit_hierarchical_nested_cv.sh --dependency 12345 --only kw_nmc_pens my_run  # wait for job 12345
#
# Two-phase submission for plateau ensembles (_pens):
#   Phase 1: submit base pipelines (skip _pens)
#     bash code/submit_hierarchical_nested_cv.sh --skip kw_nmc_pens,standalone_en_pens,en_nmc_pens my_run
#   Phase 2: after Phase 1 completes, submit _pens with dependency
#     bash code/submit_hierarchical_nested_cv.sh --only kw_nmc_pens,standalone_en_pens,en_nmc_pens --dependency <phase1_job_id> my_run
#
# Array mapping:
#     Task ID = pipeline_index * n_repeats + (repeat - 1)
#     pipeline_index: 0..N mapping to the N pipeline variants in the config.
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
    handle_signal() {
        echo "[HNCV-TASK-KILLED] array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stage2_pipeline=${PIPELINE:-unknown} repeat=${REPEAT:-unknown} signal=$1"
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

    # Read pipeline names from exported PIPELINES_STR (space-delimited).
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
    echo "Stage 1:  kw_rf k=5 (fixed, threshold calibrated)"
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
    PYTHON_CMD=(python3 "$PROJECT_DIR/code/hierarchical_nested_cv_runner.py" \
        --pipeline "$PIPELINE" \
        --repeat "$REPEAT" \
        --config "$CONFIG_FILE" \
        --name "$RUN_NAME")
    echo "Running: ${PYTHON_CMD[*]}"

    # Run with explicit exit-code capture.
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
    echo "[HNCV-TASK-DONE] array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stage2_pipeline=${PIPELINE} repeat=${REPEAT} exit=${EXIT_CODE} elapsed=${ELAPSED}s"

    exit $EXIT_CODE
fi

# ===========================================================================
# Outside SLURM: read config, set up run directory, and submit.
# ===========================================================================

# ---------------------------------------------------------------------------
# Parse arguments: [--skip pipe1,pipe2,...] [--only pipe1,pipe2,...]
#                   [--dependency job_id] [run_name] [config_file]
# ---------------------------------------------------------------------------
SKIP_PIPELINES=""
ONLY_PIPELINES=""
DEPENDENCY_JOB=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip)
            SKIP_PIPELINES="$2"
            shift 2
            ;;
        --only)
            ONLY_PIPELINES="$2"
            shift 2
            ;;
        --dependency)
            DEPENDENCY_JOB="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

RUN_NAME="${POSITIONAL_ARGS[0]:-default_run}"
CONFIG_FILE="${POSITIONAL_ARGS[1]:-config_files/server.yaml}"

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
skip = set('$SKIP_PIPELINES'.split(',')) if '$SKIP_PIPELINES' else set()
only = set('$ONLY_PIPELINES'.split(',')) if '$ONLY_PIPELINES' else set()
if only:
    pipes = [p for p in pipes if p in only]
if skip:
    pipes = [p for p in pipes if p not in skip]
if not pipes:
    raise ValueError('All pipelines were filtered out. Nothing to submit.')
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
if skip:
    print(f'SKIPPED_STR=\"{\" \".join(sorted(skip))}\"')
else:
    print('SKIPPED_STR=\"\"')
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
RUNNER_SCRIPT="$PROJECT_DIR/code/hierarchical_nested_cv_runner.py"
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
    echo "ERROR: runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

RUN_DIR=$(resolve_run_dir)
PHASE_DIR="$RUN_DIR/hierarchical_nested_cv"
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
    'stage1': 'kw_rf k=5 (fixed, threshold calibrated)',
    'stage2_pipelines': $(printf '%s\n' "${PIPELINES[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))"),
    'skipped_pipelines': '$SKIPPED_STR'.split() if '$SKIPPED_STR' else [],
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

# ---------------------------------------------------------------------------
# Split pipelines into base (Phase 1) and plateau ensemble (Phase 2).
# Plateau ensembles (_pens) need base pipeline inner_cv results to exist,
# so they run as a dependent job that starts after the base job completes.
# ---------------------------------------------------------------------------
BASE_PIPELINES=()
PENS_PIPELINES=()
for p in "${PIPELINES[@]}"; do
    case "$p" in
        *_pens) PENS_PIPELINES+=("$p") ;;
        *)      BASE_PIPELINES+=("$p") ;;
    esac
done

N_BASE=${#BASE_PIPELINES[@]}
N_PENS=${#PENS_PIPELINES[@]}
BASE_JOBS=$(( N_BASE * REPEATS_PER_PIPELINE ))
PENS_JOBS=$(( N_PENS * REPEATS_PER_PIPELINE ))

echo "========================================"
echo "Submitting hierarchical nested CV"
echo "  Run name:    $RUN_NAME"
echo "  Run dir:     $RUN_DIR"
echo "  Config file: $CONFIG_FILE"
echo "  Frozen as:   $FROZEN_CONFIG"
echo "  Stage 1:     kw_rf k=5 (fixed, threshold=0.5)"
if [[ $N_BASE -gt 0 ]]; then
    echo "  Phase 1:     ${BASE_PIPELINES[*]} (${N_BASE} pipelines, ${BASE_JOBS} jobs)"
fi
if [[ $N_PENS -gt 0 ]]; then
    echo "  Phase 2:     ${PENS_PIPELINES[*]} (${N_PENS} pipelines, ${PENS_JOBS} jobs, afterok Phase 1)"
fi
if [[ -n "$SKIPPED_STR" ]]; then
    echo "  Skipped:     $SKIPPED_STR"
fi
echo "  SLURM:       mem=${SLURM_MEM}  time=${SLURM_TIME}  cpus=${SLURM_CPUS}  max_concurrent=${SLURM_MAX_CONCURRENT}"
echo "  SLURM logs:  $SLURM_LOG_DIR"
echo "========================================"

# Build mail arguments only if a mail user is configured.
MAIL_ARGS=""
if [[ -n "$SLURM_MAIL_USER" ]]; then
    MAIL_ARGS="--mail-type=END,FAIL --mail-user=$SLURM_MAIL_USER"
fi

# Build dependency argument if specified externally.
DEP_ARGS=""
if [[ -n "$DEPENDENCY_JOB" ]]; then
    DEP_ARGS="--dependency=afterok:${DEPENDENCY_JOB}"
fi

# Helper: submit one array job for a set of pipelines.
submit_array() {
    local job_name="$1"
    local dep_args="$2"
    shift 2
    local pipes=("$@")
    local n_pipes=${#pipes[@]}
    local n_jobs=$(( n_pipes * REPEATS_PER_PIPELINE ))
    local pipes_str="${pipes[*]}"

    sbatch \
        --job-name="${job_name}" \
        --array="0-$(( n_jobs - 1 ))%${SLURM_MAX_CONCURRENT}" \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS" \
        --mem="$SLURM_MEM" \
        --time="$SLURM_TIME" \
        --output="$SLURM_LOG_DIR/hncv_%A_%a.out" \
        --error="$SLURM_LOG_DIR/hncv_%A_%a.err" \
        $MAIL_ARGS \
        $dep_args \
        --export=NONE,PROJECT_DIR="$PROJECT_DIR",RUN_NAME="$RUN_NAME",CONFIG_FILE="$FROZEN_CONFIG",REPEATS_PER_PIPELINE="$REPEATS_PER_PIPELINE",PIPELINES_STR="$pipes_str",CONDA_PREFIX_DIR="$CONDA_PREFIX_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",HOME="$HOME",USER="$USER",PATH="$PATH" \
        --parsable \
        "${BASH_SOURCE[0]}"
}

# Phase 1: base pipelines.
BASE_JOB_ID=""
if [[ $N_BASE -gt 0 ]]; then
    BASE_JOB_ID=$(submit_array "hncv_${RUN_NAME}" "$DEP_ARGS" "${BASE_PIPELINES[@]}")
    echo "Phase 1 submitted: job $BASE_JOB_ID (${N_BASE} pipelines x ${REPEATS_PER_PIPELINE} repeats = ${BASE_JOBS} tasks)"
fi

# Phase 2: plateau ensemble pipelines (depend on Phase 1).
if [[ $N_PENS -gt 0 ]]; then
    PENS_DEP=""
    if [[ -n "$BASE_JOB_ID" ]]; then
        PENS_DEP="--dependency=afterok:${BASE_JOB_ID}"
    elif [[ -n "$DEP_ARGS" ]]; then
        PENS_DEP="$DEP_ARGS"
    fi
    PENS_JOB_ID=$(submit_array "hncv_pens_${RUN_NAME}" "$PENS_DEP" "${PENS_PIPELINES[@]}")
    echo "Phase 2 submitted: job $PENS_JOB_ID (${N_PENS} pipelines x ${REPEATS_PER_PIPELINE} repeats = ${PENS_JOBS} tasks, afterok:${BASE_JOB_ID:-none})"
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Run analysis after completion with:"
echo "  python3 code/analyse_nested_cv.py --name $RUN_NAME --config $FROZEN_CONFIG --phase hierarchical_nested_cv"

exit 0
