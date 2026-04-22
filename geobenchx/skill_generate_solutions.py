import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from geobenchx.constants import RESULTS_FOLDER
from geobenchx.dataclasses import TaskSet
from geobenchx.save_chats import save_conversation_to_html
from geobenchx.skill_agent import execute_task
from geobenchx.utils import get_solution_code

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
    output_filename: str = None,
    max_steps: int = 25,
    skip_solved: bool = True,
    capture_history: bool = False,
) -> tuple[TaskSet, int, int]:
    run_folder = None
    if capture_history:
        safe_model_name = model.replace("/", "-").replace("\\", "-").replace(":", "-")
        run_folder = (
            Path(RESULTS_FOLDER)
            / Path(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{safe_model_name}_temp{temperature}")
        )
        run_folder.mkdir(parents=True, exist_ok=True)

    tasks.metadata["model"] = model
    tasks.metadata["temperature"] = temperature
    total_input = 0
    total_output = 0
    last_tool_call_ts: float | None = None

    for task in tqdm(tasks):
        print(f"Task ID: {task.task_ID}")
        print(f"Task text: {task.task_text}")
        if task.generated_solution is not None and skip_solved:
            print("Skipping task, it is alredy solved.")
            continue

        success = False
        while not success:
            try:
                _wait_for_tool_cooldown(last_tool_call_ts, TOOL_CALL_DELAY_SECONDS)
                solution, input_tokens, output_tokens, conversation_history, final_message = execute_task(
                    task.task_text,
                    temperature=temperature,
                    model=model,
                    max_steps=max_steps,
                    capture_history=capture_history,
                )
                print("=" * 30)
                print(get_solution_code(solution))
                print(
                    f"Tokens used: input tokens {sum(input_tokens)}, output_tokens {sum(output_tokens)}"
                )
                print("=" * 30)

                total_input += sum(input_tokens)
                total_output += sum(output_tokens)
                task.generated_solution = solution
                task.generated_solution_input_tokens = sum(input_tokens)
                task.generated_solution_output_tokens = sum(output_tokens)
                task.generated_solution_message = final_message

                if output_filename:
                    tasks.save_to_file(output_filename, folder=RESULTS_FOLDER)

                success = True

                if capture_history:
                    save_conversation_to_html(task, conversation_history, run_folder)
                    del conversation_history
                time.sleep(POST_TASK_DELAY_SECONDS)

            except Exception as e:
                print(repr(e))
            finally:
                last_tool_call_ts = time.time()

    print(
        f"TOTAL tokens used: total input tokens {total_input}, total output_tokens {total_output}"
    )
    tasks.metadata["total_input_tokens_for_generation"] = total_input
    tasks.metadata["total_output_tokens_for_generation"] = total_output

    if output_filename:
        tasks.save_to_file(output_filename, folder=RESULTS_FOLDER)

    return tasks, total_input, total_output
