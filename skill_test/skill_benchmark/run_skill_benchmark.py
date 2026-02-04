"""Benchmark runner that processes SKILL task sets via the standard generation loop."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geobenchx.constants import MODEL_GPT_4o
from geobenchx.dataclasses import TaskSet
from geobenchx.generate_solutions import generate_solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate through a SKILL JSON task set and run the GeoBenchX agent."
    )
    parser.add_argument(
        "--tasks-file",
        default="skill_test/dataset/skill_tasks_and_reference_solutions.json",
        help="Path to the SKILL task JSON file.",
    )
    parser.add_argument(
        "--output-filename",
        default="skill_skill_generated_solutions.json",
        help="Filename for the generated solutions JSON.",
    )
    parser.add_argument(
        "--results-folder",
        default="skill_test/skill_benchmark/results",
        help="Folder where JSON and HTML logs will be stored.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_GPT_4o,
        help="LLM model identifier (same as notebooks/Benchmarking.ipynb).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for solution generation.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum agent recursion depth.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every task even if it already contains a generated solution.",
    )
    parser.add_argument(
        "--no-history",
        dest="capture_history",
        action="store_false",
        help="Disable HTML conversation capture (enabled by default).",
    )
    parser.set_defaults(capture_history=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks_path = Path(args.tasks_file)
    task_set = TaskSet.read_from_file(tasks_path.name, folder=str(tasks_path.parent))

    task_set.metadata.setdefault(
        "notes", "Skill-oriented tasks leveraging GeoSpatialProcessingSkill wrappers."
    )
    task_set.metadata["skill_benchmark_runner"] = "skill_test.skill_benchmark.run_skill_benchmark"
    task_set.metadata["skill_benchmark_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    generate_solutions(
        tasks=task_set,
        model=args.model,
        temperature=args.temperature,
        output_filename=args.output_filename,
        max_steps=args.max_steps,
        skip_solved=not args.force,
        capture_history=args.capture_history,
        results_folder=args.results_folder,
    )


if __name__ == "__main__":
    main()
