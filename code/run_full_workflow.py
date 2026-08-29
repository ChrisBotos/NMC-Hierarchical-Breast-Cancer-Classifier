"""
Group 9.
Authors:
    Alexandros Michailidis.
    Antonie Wagner.
    Christos Botos.
    Yan Qiao.
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: run_full_workflow.py.
Description:
    Orchestrator that runs the full analysis workflow sequentially:
        1. Phase 0 - Data exploration on raw data.
        2. Phase 1 - Preprocessing (label-free region merging).
        3. Phase 0 - Data exploration on merged data (--tag merged).
        4. Comparison figure (raw vs merged).

Usage:
    python3 code/run_full_workflow.py
    python3 code/run_full_workflow.py --name my_experiment

Dependencies:
    Python >= 3.10.
"""

import argparse
import subprocess
import sys

from utils.paths import CODE_DIR, PROJECT_DIR, _find_or_create_run_dir


def build_steps(run_name):
    """Build the workflow step list for a given run name.

    The merged data input path is resolved from the run directory's
    preprocessing handoff after Phase 1 completes.

    Args:
        run_name (str): Run name passed through to all sub-scripts.

    Returns:
        list[tuple[str, list[str]]]: (description, command) pairs.
    """
    run_dir = _find_or_create_run_dir(run_name)
    merged_data = run_dir / "preprocessing" / "data" / "train_merged.tsv"

    steps = [
        (
            "Phase 0: Exploration (raw data)",
            [
                sys.executable,
                str(CODE_DIR / "data_exploration_phase.py"),
                "--name",
                run_name,
            ],
        ),
        (
            "Phase 1: Preprocessing (region merging)",
            [
                sys.executable,
                str(CODE_DIR / "preprocessing_phase.py"),
                "--name",
                run_name,
            ],
        ),
        (
            "Phase 0: Exploration (merged data)",
            [
                sys.executable,
                str(CODE_DIR / "data_exploration_phase.py"),
                "--input",
                str(merged_data),
                "--tag",
                "merged",
                "--name",
                run_name,
            ],
        ),
        (
            "Comparison: Raw vs Merged",
            [
                sys.executable,
                str(CODE_DIR / "compare_explorations.py"),
                "--name",
                run_name,
                "--label-a",
                "Raw (2834 regions)",
                "--label-b",
                "Merged (273 segments)",
            ],
        ),
    ]
    return steps


def main():
    """Run all workflow steps sequentially, stopping on first failure."""
    parser = argparse.ArgumentParser(
        description="Orchestrator: run the full analysis workflow.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default_run",
        help="Run name for the results directory (default: default_run).",
    )
    args = parser.parse_args()

    steps = build_steps(args.name)

    print("=" * 60)
    print(f"FULL WORKFLOW  (run: {args.name})")
    print("=" * 60)

    for i, (description, cmd) in enumerate(steps, 1):
        print(f"\n{'=' * 60}")
        print(f"STEP {i}/{len(steps)}: {description}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))

        if result.returncode != 0:
            print(f"\nERROR: Step {i} failed with return code {result.returncode}.")
            sys.exit(result.returncode)

        print(f"\nStep {i} complete.")

    print(f"\n{'=' * 60}")
    print("ALL WORKFLOW STEPS COMPLETE.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
