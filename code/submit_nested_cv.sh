#!/bin/bash
# Self-submitting SLURM wrapper for nested_cv_2x2_runner.py.
#
# When run outside SLURM: resolves the run directory, reads job parameters
# from the YAML config file, creates SLURM log directories, and submits
# itself as an array job.
# When run inside SLURM: maps the array task ID to (pipeline, repeat)
# and runs the Python script.
#
# Usage:
#     bash code/submit_nested_cv.sh                                              # default_run, config_files/server.yaml
#     bash code/submit_nested_cv.sh my_experiment                                # custom run name, server config
#     bash code/submit_nested_cv.sh my_experiment config_files/local.yaml        # custom run name + local config
#
# Array mapping:
#     Task ID = pipeline_index * n_repeats + (repeat - 1)
#     pipeline_index: 0=kw_nmc, 1=kw_rf, 2=en_nmc, 3=en_rf

set -euo pipefail

# Resolve project root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Arguments.
RUN_NAME="${1:-default_run}"
CONFIG_FILE="${2:-config_files/server.yaml}"

# Resolve config path relative to project root if not absolute.
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Read job parameters from the YAML config file via Python.
# ---------------------------------------------------------------------------
read_config() {
    python3 -c "
import yaml, json, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
# Print as JSON so bash can parse individual values.
print(json.dumps({
    'n_repeats': cfg['cv']['n_repeats'],
    'pipelines': cfg['pipelines']['names'],
    'mem': cfg['slurm']['mem'],
    'time': cfg['slurm']['time'],
    'cpus_per_task': cfg['slurm']['cpus_per_task'],
    'max_concurrent': cfg['slurm']['max_concurrent_jobs'],
    'mail_user': cfg['slurm'].get('mail_user', ''),
}))
"
}

CONFIG_JSON=$(read_config)

# Extract individual values from the JSON.
REPEATS_PER_PIPELINE=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_repeats'])")
SLURM_MEM=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['mem'])")
SLURM_TIME=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['time'])")
SLURM_CPUS=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['cpus_per_task'])")
SLURM_MAX_CONCURRENT=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['max_concurrent'])")
SLURM_MAIL_USER=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['mail_user'])")

# Read pipeline names into bash array.
eval "PIPELINES=($(echo "$CONFIG_JSON" | python3 -c "import sys,json; pipes=json.load(sys.stdin)['pipelines']; print(' '.join(f'\"{p}\"' for p in pipes))"))"

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

RUN_DIR=$(resolve_run_dir)
PHASE_DIR="$RUN_DIR/nested_cv_2x2"
SLURM_LOG_DIR="$PHASE_DIR/logs/slurm"

# ---------------------------------------------------------------------------
# Outside SLURM: set up and submit.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$SLURM_LOG_DIR"

    # Save config snapshot.
    SNAPSHOT_FILE="$RUN_DIR/config.json"
    TIMESTAMP=$(date -Iseconds)

    # Use python to merge into existing config.json safely.
    python3 -c "
import json, pathlib, sys
config_path = pathlib.Path('$SNAPSHOT_FILE')
config = json.loads(config_path.read_text()) if config_path.exists() else {}
config['submit_nested_cv'] = {
    'timestamp': '$TIMESTAMP',
    'run_name': '$RUN_NAME',
    'config_file': '$CONFIG_FILE',
    'repeats_per_pipeline': $REPEATS_PER_PIPELINE,
    'pipelines': $(printf '%s\n' "${PIPELINES[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))"),
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
    echo "Submitting nested CV array job"
    echo "  Run name:    $RUN_NAME"
    echo "  Run dir:     $RUN_DIR"
    echo "  Config file: $CONFIG_FILE"
    echo "  Jobs:        $TOTAL_JOBS (${N_PIPELINES} pipelines x ${REPEATS_PER_PIPELINE} repeats)"
    echo "  SLURM:       mem=${SLURM_MEM}  time=${SLURM_TIME}  cpus=${SLURM_CPUS}  max_concurrent=${SLURM_MAX_CONCURRENT}"
    echo "  SLURM logs:  $SLURM_LOG_DIR"
    echo "========================================"

    # Build mail arguments only if a mail user is configured.
    MAIL_ARGS=""
    if [[ -n "$SLURM_MAIL_USER" ]]; then
        MAIL_ARGS="--mail-type=BEGIN,END,FAIL,ARRAY_TASKS --mail-user=$SLURM_MAIL_USER"
    fi

    # Capture the array job ID so the analysis job can depend on it.
    ARRAY_JOB_ID=$(sbatch \
        --job-name="ncv_${RUN_NAME}" \
        --array="0-$(( TOTAL_JOBS - 1 ))%${SLURM_MAX_CONCURRENT}" \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS" \
        --mem="$SLURM_MEM" \
        --time="$SLURM_TIME" \
        --output="$SLURM_LOG_DIR/nested_cv_%A_%a.out" \
        --error="$SLURM_LOG_DIR/nested_cv_%A_%a.err" \
        $MAIL_ARGS \
        --export=ALL,RUN_NAME="$RUN_NAME",CONFIG_FILE="$CONFIG_FILE",REPEATS_PER_PIPELINE="$REPEATS_PER_PIPELINE" \
        --parsable \
        "${BASH_SOURCE[0]}")

    echo "Array job submitted: $ARRAY_JOB_ID"

    # Submit a dependent analysis job that runs after all array tasks complete.
    ANALYSIS_JOB_ID=$(sbatch \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        --job-name="ncv_analysis_${RUN_NAME}" \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=4G \
        --time="0-00:30:00" \
        --output="$SLURM_LOG_DIR/nested_cv_analysis_%j.out" \
        --error="$SLURM_LOG_DIR/nested_cv_analysis_%j.err" \
        $MAIL_ARGS \
        --parsable \
        --wrap="source ~/miniconda3/bin/activate tb_310 && python3 $PROJECT_DIR/code/analyse_nested_cv.py --name $RUN_NAME --config $CONFIG_FILE")

    echo "Analysis job submitted: $ANALYSIS_JOB_ID (depends on $ARRAY_JOB_ID)"

    exit 0
fi

# ---------------------------------------------------------------------------
# Inside SLURM: activate environment and run one (pipeline, repeat) job.
# ---------------------------------------------------------------------------
source ~/miniconda3/bin/activate tb_310

# Re-read pipeline names from config (exported env vars do not include arrays).
eval "PIPELINES=($(python3 -c "
import yaml, json
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(' '.join(f'\"{p}\"' for p in cfg['pipelines']['names']))
"))"
N_PIPELINES=${#PIPELINES[@]}
TOTAL_JOBS=$(( N_PIPELINES * REPEATS_PER_PIPELINE ))

PIPELINE_IDX=$(( SLURM_ARRAY_TASK_ID / REPEATS_PER_PIPELINE ))
REPEAT=$(( SLURM_ARRAY_TASK_ID % REPEATS_PER_PIPELINE + 1 ))
PIPELINE="${PIPELINES[$PIPELINE_IDX]}"

JOB_START=$(date +%s)
echo "========================================"
echo "Job ${SLURM_ARRAY_TASK_ID} of $(( TOTAL_JOBS - 1 ))"
echo "Pipeline: ${PIPELINE}"
echo "Repeat:   ${REPEAT} / ${REPEATS_PER_PIPELINE}"
echo "Config:   ${CONFIG_FILE}"
echo "Run:      ${RUN_NAME}"
echo "Node:     $(hostname)"
echo "Start:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python3 "$PROJECT_DIR/code/nested_cv_2x2_runner.py" \
    --pipeline "$PIPELINE" \
    --repeat "$REPEAT" \
    --config "$CONFIG_FILE" \
    --name "$RUN_NAME"

JOB_END=$(date +%s)
ELAPSED=$(( JOB_END - JOB_START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))
echo "========================================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Elapsed:  ${HOURS}h ${MINS}m ${SECS}s"
echo "========================================"
