"""Shared constants for the SKILL benchmark runner."""

from __future__ import annotations

from pathlib import Path

from geobenchx.constants import (  # Reuse model identifiers and enums from the main benchmark.
    MODEL_CLAUDE,
    MODEL_CLAUDE_ADV3,
    MODEL_CLAUDE_ADV4,
    MODEL_CLAUDE_mini,
    MODEL_GEMINI,
    MODEL_GEMINI_ADV,
    MODEL_GPT_41,
    MODEL_GPT_4o,
    MODEL_GPT_mini,
    MODEL_O3,
    MODEL_O4,
    MODEL_SHER_LOCKER,
    MODEL_SHER_LOCKER_4mini,
    MODEL_SHER_LOCKER_4o,
    MODEL_SHER_LOCKER_GEMINI_FLASH,
    NO_LABEL,
    ScoreValues,
    TaskLabels,
)

SKILL_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = SKILL_ROOT / "dataset"
RESULTS_FOLDER = SKILL_ROOT / "results"
SKILL_SCRATCH_PATH = SKILL_ROOT / "scratch"

TASKS_FILENAME = "skill_tasks_and_reference_solutions.json"
GENERATED_SOLUTIONS_FILENAME = "skill_generated_solutions.json"

DEFAULT_MODEL = MODEL_SHER_LOCKER_4o
DEFAULT_TEMPERATURE = 0.0

# Model registry for UI pickers / notebooks
MODEL_REGISTRY = {
    "GPT-4o (Aug-2024)": MODEL_GPT_4o,
    "GPT-4.1 (Apr-2025)": MODEL_GPT_41,
    "GPT-4.1 Mini (Apr-2025)": MODEL_GPT_mini,
    "o3-mini (Jan-2025)": MODEL_O3,
    "o4-mini (Apr-2025)": MODEL_O4,
    "Gemini 2.5 Flash": MODEL_GEMINI,
    "Gemini 2.5 Pro Preview": MODEL_GEMINI_ADV,
    "Claude 3.5 Sonnet (Oct-2024)": MODEL_CLAUDE,
    "Claude 3.5 Haiku (Oct-2024)": MODEL_CLAUDE_mini,
    "Claude 3.7 Sonnet (Feb-2025)": MODEL_CLAUDE_ADV3,
    "Claude Sonnet 4 (May-2025)": MODEL_CLAUDE_ADV4,
    "Sher Locker GPT-4.1 Nano": MODEL_SHER_LOCKER,
    "gpt-4o": MODEL_SHER_LOCKER_4o,
    "Sher Locker Gemini Flash": MODEL_SHER_LOCKER_GEMINI_FLASH,
    "Sher Locker GPT-4o Mini": MODEL_SHER_LOCKER_4mini,
}
AVAILABLE_MODELS = list(MODEL_REGISTRY.values())

# Cooldown timings mirror the main GeoBench agent defaults, but keeping them here
# allows the SKILL harness to be tuned independently.
TOOL_CALL_DELAY_SECONDS = 20.0
POST_TASK_DELAY_SECONDS = 30.0

__all__ = [
    "DATA_FOLDER",
    "RESULTS_FOLDER",
    "SKILL_SCRATCH_PATH",
    "TASKS_FILENAME",
    "GENERATED_SOLUTIONS_FILENAME",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "MODEL_REGISTRY",
    "AVAILABLE_MODELS",
    "TOOL_CALL_DELAY_SECONDS",
    "POST_TASK_DELAY_SECONDS",
    "MODEL_GPT_4o",
    "MODEL_GPT_41",
    "MODEL_GPT_mini",
    "MODEL_O3",
    "MODEL_O4",
    "MODEL_GEMINI",
    "MODEL_GEMINI_ADV",
    "MODEL_CLAUDE",
    "MODEL_CLAUDE_mini",
    "MODEL_CLAUDE_ADV3",
    "MODEL_CLAUDE_ADV4",
    "MODEL_SHER_LOCKER",
    "MODEL_SHER_LOCKER_4o",
    "MODEL_SHER_LOCKER_GEMINI_FLASH",
    "MODEL_SHER_LOCKER_4mini",
    "ScoreValues",
    "TaskLabels",
    "NO_LABEL",
]
