#!/bin/bash
#SBATCH --job-name=nested_cv_2x2
#SBATCH --array=0-199%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/slurm/nested_cv_%A_%a.out
#SBATCH --error=results/logs/slurm/nested_cv_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=hcty03@gmail.com

# SLURM submission script for nested_cv_2x2_runner.py.
# Submits 200 jobs (4 pipelines x 50 repeats) as an array.
#
# Usage:
#     sbatch code/submit_nested_cv.sh
#     sbatch code/submit_nested_cv.sh trial   # smaller grid for testing
#
# Array mapping:
#     Task ID = pipeline_index * 50 + (repeat - 1)
#     pipeline_index: 0=kw_nmc, 1=kw_rf, 2=en_nmc, 3=en_rf

set -euo pipefail

# Activate conda environment.
source ~/miniconda3/bin/activate tb_310

# Resolve project root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create SLURM log directory if needed.
mkdir -p "$PROJECT_DIR/results/logs/slurm"

# Map array task ID to (pipeline, repeat).
REPEATS_PER_PIPELINE=50
PIPELINES=("kw_nmc" "kw_rf" "en_nmc" "en_rf")
PIPELINE_IDX=$(( SLURM_ARRAY_TASK_ID / REPEATS_PER_PIPELINE ))
REPEAT=$(( SLURM_ARRAY_TASK_ID % REPEATS_PER_PIPELINE + 1 ))
PIPELINE="${PIPELINES[$PIPELINE_IDX]}"

# Forward any extra arguments (e.g. --config production).
CONFIG="${1:-production}"

# Print job info and start time for progress tracking.
JOB_START=$(date +%s)
echo "========================================"
echo "Job ${SLURM_ARRAY_TASK_ID} of 199"
echo "Pipeline: ${PIPELINE}"
echo "Repeat:   ${REPEAT} / ${REPEATS_PER_PIPELINE}"
echo "Config:   ${CONFIG}"
echo "Node:     $(hostname)"
echo "Start:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python3 "$PROJECT_DIR/code/nested_cv_2x2_runner.py" \
    --pipeline "$PIPELINE" \
    --repeat "$REPEAT" \
    --config "$CONFIG"

# Print elapsed time on completion.
JOB_END=$(date +%s)
ELAPSED=$(( JOB_END - JOB_START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))
echo "========================================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Elapsed:  ${HOURS}h ${MINS}m ${SECS}s"
echo "========================================"
