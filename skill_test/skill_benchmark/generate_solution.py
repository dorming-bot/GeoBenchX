"""Isolated generate_solutions implementation for SKILL benchmarking."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geobenchx.agent import execute_task
from geobenchx.dataclasses import TaskSet
from geobenchx.save_chats import save_conversation_to_html
from geobenchx.utils import get_solution_code

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOOL_CALL_DELAY_SECONDS = 20.0
POST_TASK_DELAY_SECONDS = 60.0


def _wait_for_tool_cooldown(last_call_ts: float | None, min_interval: float) -> None:
    if last_call_ts is None:
        return
    elapsed = time.time() - last_call_ts
    remaining = min_interval - elapsed
    if remaining > 0:
        time.sleep(remaining)


def generate_solutions(
    tasks: TaskSet,
    model: str,
    temperature: float,
    output_filename: str | None = None,
    max_steps: int = 25,
    skip_solved: bool = True,
    capture_history: bool = True,
    results_folder: str | Path | None = None,
) -> Tuple[TaskSet, int]:
    """Run the GeoBenchX agent across all tasks and store outputs inside skill_test."""
    results_dir = Path(results_folder) if results_folder else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    run_folder: Path | None = None
    if capture_history:
        safe_model_name = model.replace("/", "-").replace("\\", "-").replace(":", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = results_dir / f"{timestamp}_{safe_model_name}_temp{temperature}"
        run_folder.mkdir(parents=True, exist_ok=True)

    tasks.metadata["model"] = model
    tasks.metadata["temperature"] = temperature

    total_input = 0
    total_output = 0
    last_tool_call_ts: float | None = None

    for task in tasks:
        print(f"Task ID: {task.task_ID}")
        print(f"Task text: {task.task_text}")
        if task.generated_solution is not None and skip_solved:
            print("Skipping task, it is already solved.")
            continue

        success = False
        while not success:
            try:
                _wait_for_tool_cooldown(last_tool_call_ts, TOOL_CALL_DELAY_SECONDS)
                (
                    solution,
                    input_tokens,
                    output_tokens,
                    conversation_history,
                    final_message,
                ) = execute_task(
                    task.task_text,
                    temperature=temperature,
                    model=model,
                    max_steps=max_steps,
                    capture_history=capture_history,
                )
                print("=" * 30)
                print(get_solution_code(solution))
                print(
                    f"Tokens used: input {sum(input_tokens)}, output {sum(output_tokens)}"
                )
                print("=" * 30)

                total_input += sum(input_tokens)
                total_output += sum(output_tokens)

                task.generated_solution = solution
                task.generated_solution_input_tokens = sum(input_tokens)
                task.generated_solution_output_tokens = sum(output_tokens)
                task.generated_solution_message = final_message

                if output_filename:
                    tasks.save_to_file(output_filename, folder=str(results_dir))

                if capture_history and conversation_history is not None and run_folder:
                    save_conversation_to_html(task, conversation_history, run_folder)

                success = True
                time.sleep(POST_TASK_DELAY_SECONDS)
            except Exception as exc:
                print(repr(exc))
            finally:
                last_tool_call_ts = time.time()

    print(
        f"TOTAL tokens used: input {total_input}, output {total_output}"
    )
    tasks.metadata["total_input_tokens_for_generation"] = total_input
    tasks.metadata["total_output_tokens_for_generation"] = total_output

    if output_filename:
        tasks.save_to_file(output_filename, folder=str(results_dir))

    return tasks, total_input, total_output


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SKILL benchmarking using the isolated generator."
    )
    parser.add_argument(
        "--tasks-file",
        default="skill_test/dataset/skill_tasks_and_reference_solutions.json",
        help="Path to the JSON task set.",
    )
    parser.add_argument(
        "--results-folder",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where JSON/HTML outputs are stored.",
    )
    parser.add_argument(
        "--output-filename",
        default="skill_generated_solutions.json",
        help="Filename for the generated solutions JSON.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help="Model identifier passed to the agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum tool-calling depth.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks even if they already have generated solutions.",
    )
    parser.add_argument(
        "--no-history",
        dest="capture_history",
        action="store_false",
        help="Disable HTML capture.",
    )
    parser.set_defaults(capture_history=True)
    args = parser.parse_args()

    tasks_path = Path(args.tasks_file)
    task_set = TaskSet.read_from_file(tasks_path.name, folder=str(tasks_path.parent))

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
    _cli()
