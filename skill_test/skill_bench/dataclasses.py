"""Pydantic data models for the SKILL benchmark datasets and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel, Field

from skill_test.skill_bench.constants import (
    DATA_FOLDER,
    GENERATED_SOLUTIONS_FILENAME,
    RESULTS_FOLDER,
    ScoreValues,
    TaskLabels,
)


class Step(BaseModel):
    """Represents a single SKILL/tool invocation."""

    function_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    comment: Optional[str] = None


class Solution(BaseModel):
    """A solution is a sequence of tool calls and optional comments."""

    steps: List[Step] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterable[Step]:
        return iter(self.steps)


class Task(BaseModel):
    """Container for a single SKILL task, including reference and generated solutions."""

    task_ID: str
    task_text: str

    task_labels: List[TaskLabels] = Field(default_factory=list)

    reference_solution_description: Optional[str] = None
    reference_solutions: List[Solution] = Field(default_factory=list)

    generated_solution: Optional[Solution] = None
    generated_solution_input_tokens: Optional[int] = None
    generated_solution_output_tokens: Optional[int] = None
    generated_solution_message: Optional[str] = None

    match_reasoning_LLM: Optional[str] = None
    match_score_LLM: Optional[ScoreValues] = None
    match_reasoning_Human: Optional[str] = None
    match_score_Human: Optional[ScoreValues] = None

    class Config:
        extra = "ignore"


class TaskSet(BaseModel):
    """Simple wrapper that combines benchmark metadata with an ordered list of tasks."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    tasks: List[Task] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: Union[int, slice]) -> Union[Task, List[Task]]:
        return self.tasks[index]

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        return next((task for task in self.tasks if task.task_ID == task_id), None)

    def save_to_file(
        self,
        filename: Optional[str] = None,
        *,
        folder: Union[str, Path, None] = None,
        indent: int = 4,
    ) -> Path:
        """Serialize the task set to disk, mirroring the source JSON layout."""
        target_folder = Path(folder) if folder else Path(RESULTS_FOLDER)
        target_folder.mkdir(parents=True, exist_ok=True)
        path = target_folder / (filename or GENERATED_SOLUTIONS_FILENAME)
        payload = self.model_dump(mode="python", exclude_none=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
        return path

    @classmethod
    def read_from_file(
        cls,
        filename: str,
        *,
        folder: Union[str, Path, None] = None,
    ) -> "TaskSet":
        """Load a TaskSet from disk."""
        source_folder = Path(folder) if folder else Path(DATA_FOLDER)
        path = source_folder / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
