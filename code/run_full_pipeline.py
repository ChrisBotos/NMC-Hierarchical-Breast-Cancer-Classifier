"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: run_full_pipeline.py.
Description:
    Orchestrator that runs the full analysis pipeline sequentially:
        1. Phase 0 — Data exploration on raw data.
        2. Phase 1 — Preprocessing (label-free region merging).
        3. Phase 0 — Data exploration on merged data (--tag merged).
        4. Comparison figure (raw vs merged).

Usage:
    python3 code/run_full_pipeline.py

Dependencies:
    Python >= 3.10.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MERGED_DATA = PROJECT_DIR / "results" / "data" / "preprocessing_phase" / "train_merged.tsv"

# Pipeline steps: (description, command list).
STEPS = [
    (
        "Phase 0: Exploration (raw data)",
        [sys.executable, str(SCRIPT_DIR / "data_exploration_phase.py")],
    ),
    (
        "Phase 1: Preprocessing (region merging)",
        [sys.executable, str(SCRIPT_DIR / "preprocessing_phase.py")],
    ),
    (
        "Phase 0: Exploration (merged data)",
        [
            sys.executable, str(SCRIPT_DIR / "data_exploration_phase.py"),
            "--input", str(MERGED_DATA),
            "--tag", "merged",
        ],
    ),
    (
        "Comparison: Raw vs Merged",
        [
            sys.executable, str(SCRIPT_DIR / "compare_explorations.py"),
            "--label-a", "Raw (2834 regions)",
            "--label-b", "Merged (273 segments)",
        ],
    ),
]


def main():
    """Run all pipeline steps sequentially, stopping on first failure."""
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)

    for i, (description, cmd) in enumerate(STEPS, 1):
        print(f"\n{'=' * 60}")
        print(f"STEP {i}/{len(STEPS)}: {description}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))

        if result.returncode != 0:
            print(f"\nERROR: Step {i} failed with return code {result.returncode}.")
            sys.exit(result.returncode)

        print(f"\nStep {i} complete.")

    print(f"\n{'=' * 60}")
    print("ALL PIPELINE STEPS COMPLETE.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
