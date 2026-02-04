import os
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Ensure the repository root is importable when running this module directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from skill_test.skill_bench.constants import (  # noqa: E402
    MODEL_CLAUDE,
    MODEL_CLAUDE_ADV3,
    MODEL_CLAUDE_ADV4,
    MODEL_CLAUDE_mini,
    MODEL_GEMINI,
    MODEL_GEMINI_ADV,
    MODEL_GPT_4o,
    MODEL_GPT_41,
    MODEL_GPT_mini,
    MODEL_O3,
    MODEL_O4,
    MODEL_SHER_LOCKER,
    MODEL_SHER_LOCKER_4mini,
    MODEL_SHER_LOCKER_4o,
    MODEL_SHER_LOCKER_GEMINI_FLASH,
)
from skill_test.skill.calculate_polygon_areas_statements.calculate_polygon_areas import (  # noqa: E402
    calculate_polygon_areas,
)
from skill_test.skill.create_dissolved_buffer_statements.create_dissolved_buffer import (  # noqa: E402
    create_dissolved_buffer,
)
from skill_test.skill.get_centroids_statements.get_centroids import (  # noqa: E402
    get_centroids,
)
from skill_test.skill.get_raster_path_statements.get_raster_path import (  # noqa: E402
    get_raster_path,
)
from skill_test.skill.get_raster_description_statements.get_raster_description import (  # noqa: E402
    get_raster_description,
)
from skill_test.skill.load_geodata_statements.load_geodata import (  # noqa: E402
    load_geodata,
)
from skill_test.skill_bench.prompts import RULES_PROMPT, SYSTEM_PROMPT  # noqa: E402

_ = load_dotenv(find_dotenv())


class State(TypedDict):
    """Local state schema for the SKILL agent graph."""
    data_store: Dict[str, Any]
    image_store: List[Dict[str, Any]]
    html_store: List[Dict[str, Any]]
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
    visualize: bool

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SHER_LOCKER_API_KEY = os.getenv("SHER_LOCKER_API_KEY")
DEFAULT_SHER_OPENAI_BASE_URL = "https://sher.locker/openai/v1/"
DEFAULT_SHER_GOOGLE_BASE_URL = "https://sher.locker/google/v1beta/"
SHER_LOCKER_OPENAI_BASE_URL = os.getenv("SHER_LOCKER_OPENAI_BASE_URL", DEFAULT_SHER_OPENAI_BASE_URL)
SHER_LOCKER_GOOGLE_BASE_URL = os.getenv("SHER_LOCKER_GOOGLE_BASE_URL", DEFAULT_SHER_GOOGLE_BASE_URL)
SHER_LOCKER_BASE_URL = os.getenv("SHER_LOCKER_BASE_URL")


def _resolve_sher_locker_base_url(model: str) -> str:
    model_lower = model.lower()
    if "gemini" in model_lower:
        return SHER_LOCKER_GOOGLE_BASE_URL or SHER_LOCKER_BASE_URL or DEFAULT_SHER_GOOGLE_BASE_URL
    return SHER_LOCKER_OPENAI_BASE_URL or SHER_LOCKER_BASE_URL or DEFAULT_SHER_OPENAI_BASE_URL


def _build_skill_tools() -> List[StructuredTool]:
    """Create StructuredTool objects for all SKILL functions."""

    def _tool(func, name: str, description: str) -> StructuredTool:
        return StructuredTool.from_function(func=func, name=name, description=description)

    return [
        _tool(
            calculate_polygon_areas,
            "calculate_polygon_areas",
            "Measure polygon areas (square kilometers) with automatic UTM selection and store the projected GeoDataFrame.",
        ),
        _tool(
            create_dissolved_buffer,
            "create_dissolved_buffer",
            "Build metric buffers around geometries, optionally dissolve by attribute, and save the results plus preview imagery.",
        ),
        _tool(
            get_centroids,
            "get_centroids",
            "Create centroids from polygon geometries, export shapefiles/TIFFs, and store summary metadata.",
        ),
        _tool(
            load_geodata,
            "load_geodata",
            "Load a vector dataset from the catalog and place the GeoDataFrame into the shared state data store.",
        ),
        _tool(
            get_raster_path,
            "get_raster_path",
            "Resolve raster dataset names to normalized GeoTIFF paths or ZIP URIs for downstream raster tools.",
        ),
        _tool(
            get_raster_description,
            "get_raster_description",
            "Summarize raster metadata and band statistics for a supplied raster file path.",
        ),
    ]


TOOLS: List[StructuredTool] = _build_skill_tools()


def _build_llm(model: str, temperature: float):
    """Return an initialized chat model instance based on the requested provider."""
    if model in [MODEL_CLAUDE, MODEL_CLAUDE_mini, MODEL_CLAUDE_ADV3, MODEL_CLAUDE_ADV4]:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required to use Claude models.")
        return ChatAnthropic(model=model, temperature=temperature)
    if model in [MODEL_GPT_4o, MODEL_GPT_41, MODEL_GPT_mini]:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required to use OpenAI models.")
        return ChatOpenAI(model=model, temperature=temperature)
    if model in [MODEL_O3, MODEL_O4]:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required to use OpenAI o-series models.")
        return ChatOpenAI(model=model, temperature=None)
    if model in [MODEL_GEMINI, MODEL_GEMINI_ADV]:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required to use Gemini models.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    if model in [MODEL_SHER_LOCKER, MODEL_SHER_LOCKER_4o, MODEL_SHER_LOCKER_GEMINI_FLASH, MODEL_SHER_LOCKER_4mini]:
        if not SHER_LOCKER_API_KEY:
            raise ValueError("SHER_LOCKER_API_KEY is required to use Sher Locker relay models.")
        base_url = _resolve_sher_locker_base_url(model)
        return ChatOpenAI(model=model, temperature=temperature, api_key=SHER_LOCKER_API_KEY, base_url=base_url)
    raise ValueError(f"Unsupported model '{model}'.")


def _extract_text_from_message(message_content) -> Optional[str]:
    """Normalize LLM message payloads (list/dict) into a plain string."""
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "\n".join(parts).strip()
        return text or None
    return None


def execute_task(
    task_text: str,
    *,
    temperature: float = 0.0,
    model: str = MODEL_GPT_4o,
    max_steps: int = 15,
    capture_history: bool = False,
) -> Dict[str, object]:
    """
    Execute a SKILL benchmark task with a ReAct-style agent.

    Returns a dictionary that includes tool-call steps, token usage, captured conversation,
    and the final assistant response. Dataclass integration will be handled separately.
    """
    conversation_history: Optional[List[Dict[str, str]]] = [] if capture_history else None
    solution_steps: List[Dict[str, Dict]] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []

    llm = _build_llm(model=model, temperature=temperature)
    graph = create_react_agent(
        llm,
        tools=TOOLS,
        state_schema=State,
        state_modifier=SYSTEM_PROMPT + RULES_PROMPT,
    )

    inputs = {
        "messages": [("user", task_text)],
        "data_store": {},
        "image_store": [],
        "html_store": [],
        "visualize": True,
    }
    config = {"max_concurrency": 1, "recursion_limit": max_steps}
    final_message: Optional[str] = None

    try:
        for state in graph.stream(inputs, stream_mode="values", config=config):
            message = state["messages"][-1]
            if hasattr(message, "usage_metadata") and message.usage_metadata:
                input_tokens.append(message.usage_metadata.get("input_tokens", 0))
                output_tokens.append(message.usage_metadata.get("output_tokens", 0))

            if hasattr(message, "tool_calls") and message.tool_calls:
                for call in message.tool_calls:
                    solution_steps.append({"function_name": call.get("name"), "arguments": call.get("args", {})})
            elif getattr(message, "type", None) == "ai":
                text = _extract_text_from_message(message.content)
                if text:
                    final_message = text

            if capture_history and conversation_history is not None:
                conversation_history.append({"type": getattr(message, "type", "unknown"), "content": message.content})

            if capture_history and conversation_history is not None:
                if "image_store" in state and state["image_store"]:
                    for img in state["image_store"]:
                        conversation_history.append(
                            {
                                "type": "image",
                                "content": img.get("base64", ""),
                                "description": img.get("description", "Visualization"),
                            }
                        )
                    state["image_store"].clear()
                if "html_store" in state and state["html_store"]:
                    for html_item in state["html_store"]:
                        conversation_history.append(
                            {
                                "type": "interactive_map",
                                "content": html_item.get("html", ""),
                                "description": html_item.get("description", "Interactive Map"),
                            }
                        )
                    state["html_store"].clear()
    except GraphRecursionError as exc:
        final_message = (final_message or "") + f"\n[Agent stopped: {exc}]"

    return {
        "steps": solution_steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "conversation_history": conversation_history,
        "final_message": final_message,
    }
