"""Batch runner that generates solutions for SKILL benchmark tasks."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from skill_test.skill_bench.agent import execute_task
from skill_test.skill_bench.constants import (
    GENERATED_SOLUTIONS_FILENAME,
    POST_TASK_DELAY_SECONDS,
    RESULTS_FOLDER,
    TOOL_CALL_DELAY_SECONDS,
)
from skill_test.skill_bench.dataclasses import Solution, Step, TaskSet
from skill_test.skill_bench.html import save_task_html


def _wait_for_tool_cooldown(last_call_ts: Optional[float], min_interval: float) -> None:
    if last_call_ts is None:
        return
    elapsed = time.time() - last_call_ts
    remaining = min_interval - elapsed
    if remaining > 0:
        time.sleep(remaining)


def generate_solutions(
    tasks: TaskSet,
    *,
    model: str,
    temperature: float,
    output_filename: Optional[str] = GENERATED_SOLUTIONS_FILENAME,
    max_steps: int = 15,
    skip_solved: bool = True,
    capture_history: bool = False,
    results_folder: Optional[Path] = None,
) -> Tuple[TaskSet, int, int, Path]:
    """
    Iterate through each task, call the agent, and persist the generated solutions.

    Returns the updated TaskSet along with aggregate input/output token counts.
    """
    results_dir = Path(results_folder) if results_folder else Path(RESULTS_FOLDER)
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "-").replace("\\", "-").replace(":", "-")
    run_dir = results_dir / f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M')}_{safe_model}_temp{temperature}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks.metadata.update(
        {
            "model": model,
            "temperature": temperature,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    )

    total_input_tokens = 0
    total_output_tokens = 0
    last_tool_call_ts: Optional[float] = None

    for task in tqdm(tasks, desc="Generating SKILL solutions"):
        if task.generated_solution is not None and skip_solved:
            continue

        print(f"\n=== Task {task.task_ID} ===")
        print(task.task_text)
        _wait_for_tool_cooldown(last_tool_call_ts, TOOL_CALL_DELAY_SECONDS)
        result = execute_task(
            task.task_text,
            temperature=temperature,
            model=model,
            max_steps=max_steps,
            capture_history=capture_history,
        )
        last_tool_call_ts = time.time()

        solution_steps = [
            Step(function_name=step["function_name"], arguments=step.get("arguments", {}))
            for step in result.get("steps", [])
        ]
        task.generated_solution = Solution(steps=solution_steps)

        for idx, step in enumerate(solution_steps, start=1):
            print(f"  Step {idx}: {step.function_name} -> {step.arguments}")

        input_total = sum(result.get("input_tokens", []))
        output_total = sum(result.get("output_tokens", []))
        total_input_tokens += input_total
        total_output_tokens += output_total

        task.generated_solution_input_tokens = input_total
        task.generated_solution_output_tokens = output_total
        task.generated_solution_message = result.get("final_message")
        if task.generated_solution_message:
            print(f"  Final message: {task.generated_solution_message}")
        print(f"  Tokens - input: {input_total}, output: {output_total}")

        try:
            html_path = save_task_html(
                task,
                run_folder=run_dir,
                conversation_history=result.get("conversation_history"),
            )
            print(f"  HTML summary saved to: {html_path.name}")
        except Exception as exc:
            print(f"  Warning: unable to write HTML for {task.task_ID}: {exc}")

        if output_filename:
            tasks.save_to_file(output_filename, folder=run_dir)

        time.sleep(POST_TASK_DELAY_SECONDS)

    tasks.metadata["total_input_tokens_for_generation"] = total_input_tokens
    tasks.metadata["total_output_tokens_for_generation"] = total_output_tokens

    if output_filename:
        tasks.save_to_file(output_filename, folder=run_dir)

    return tasks, total_input_tokens, total_output_tokens, run_dir
