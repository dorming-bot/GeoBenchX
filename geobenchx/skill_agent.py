import os
import base64
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
try:
    from IPython.display import Image as IPyImage, display as ipy_display
except Exception:  # pragma: no cover - fallback for non-notebook runtimes
    IPyImage = None
    ipy_display = None

from geobenchx.constants import (
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
    MODEL_SHER_LOCKER_GPT5.4,
)
from geobenchx.dataclasses import Solution, Step
from geobenchx.prompts import RULES_PROMPT, SYSTEM_PROMPT
from skill.calculate_polygon_areas_statements.calculate_polygon_areas import (
    calculate_polygon_areas,
)
from skill.calculate_raster_selection_area_statements.calculate_raster_selection_area import (
    calculate_raster_selection_area,
)
from skill.classify_raster_zones_statements.classify_raster_zones import (
    classify_raster_zones,
)
from skill.convert_dem_to_slope_statements.convert_dem_to_slope import (
    convert_dem_to_slope,
)
from skill.create_dissolved_buffer_statements.create_dissolved_buffer import (
    create_dissolved_buffer,
)
from skill.extract_raster_boundary_statements.extract_raster_boundary import (
    extract_raster_boundary,
)
from skill.generate_aspect_map_statements.generate_aspect_map import (
    generate_aspect_map,
)
from skill.generate_profile_curvature_map_statements.generate_profile_curvature_map import (
    generate_profile_curvature_map,
)
from skill.get_centroids_statements.get_centroids import get_centroids
from skill.get_raster_description_statements.get_raster_description import (
    get_raster_description,
)
from skill.get_raster_path_statements.get_raster_path import get_raster_path
from skill.get_unique_values_statements.get_unique_values import get_unique_values
from skill.get_values_from_raster_with_geometries_statements.get_values_from_raster_with_geometries import (
    get_values_from_raster_with_geometries,
)
from skill.filter_categorical_statements.filter_categorical import filter_categorical
from skill.filter_numerical_statements.filter_numerical import filter_numerical
from skill.filter_points_by_raster_values_statements.filter_points_by_raster_values import (
    filter_points_by_raster_values,
)
from skill.calculate_column_statistics_statements.calculate_column_statistics import (
    calculate_column_statistics,
)
from skill.calculate_columns_statements.calculate_columns import calculate_columns
from skill.create_statistical_chart_statements.create_statistical_chart import (
    create_statistical_chart,
)
from skill.reproject_vector_data_statements.reproject_vector_data import (
    reproject_vector_data,
)
from skill.convert_csv_to_shapefile_statements.convert_csv_to_shapefile import (
    convert_csv_to_shapefile,
)
from skill.calculate_line_direction_rose_statements.calculate_line_direction_rose import (
    calculate_line_direction_rose,
)
from skill.load_data_statements.load_data import load_data
from skill.load_geodata_statements.load_geodata import load_geodata
from skill.make_choropleth_map_statements.make_choropleth_map import (
    make_choropleth_map,
)
from skill.make_heatmap_statements.make_heatmap import make_heatmap
from skill.make_bivariate_map_statements.make_bivariate_map import make_bivariate_map
from skill.merge_dataframes_statements.merge_dataframes import merge_dataframes
from skill.create_point_kernel_density_map_statements.create_point_kernel_density_map import (
    create_point_kernel_density_map,
)
from skill.rasterize_vector_to_match_raster_statements.rasterize_vector_to_match_raster import (
    rasterize_vector_to_match_raster,
)
from skill.scale_column_by_value_statements.scale_column_by_value import (
    scale_column_by_value,
)
from skill.analyze_raster_overlap_statements.analyze_raster_overlap import (
    analyze_raster_overlap,
)
from skill.select_features_by_spatial_relationship_statements.select_features_by_spatial_relationship import (
    select_features_by_spatial_relationship,
)
from skill.calculate_line_lengths_statements.calculate_line_lengths import (
    calculate_line_lengths,
)
from skill.analyze_vector_overlap_statements.analyze_vector_overlap import (
    analyze_vector_overlap,
)
from skill.perform_vector_topology_operation_statements.perform_vector_topology_operation import (
    perform_vector_topology_operation,
)
from skill.calculate_nearest_distances_statements.calculate_nearest_distances import (
    calculate_nearest_distances,
)
from skill.calculate_standard_directional_ellipse_statements.calculate_standard_directional_ellipse import (
    calculate_standard_directional_ellipse,
)
from skill.visualize_geographies_statements.visualize_geographies import (
    visualize_geographies,
)
from skill.plot_contour_lines_statements.plot_contour_lines import plot_contour_lines
from skill.generate_contours_display_statements.generate_contours_display import (
    generate_contours_display,
)
from skill.reject_task_statements.reject_task import reject_task

_ = load_dotenv(find_dotenv())

_sher_locker_api_key = os.getenv("SHER_LOCKER_API_KEY")
_DEFAULT_SHER_OPENAI_BASE_URL = "https://sher.locker/openai/v1/"
_DEFAULT_SHER_GOOGLE_BASE_URL = "https://sher.locker/google/v1beta/"
_sher_locker_openai_base_url = os.getenv(
    "SHER_LOCKER_OPENAI_BASE_URL", _DEFAULT_SHER_OPENAI_BASE_URL
)
_sher_locker_google_base_url = os.getenv(
    "SHER_LOCKER_GOOGLE_BASE_URL", _DEFAULT_SHER_GOOGLE_BASE_URL
)
_sher_locker_base_url = os.getenv("SHER_LOCKER_BASE_URL")


def _resolve_sher_locker_base_url(model: str) -> str:
    model_lower = model.lower()
    if "gemini" in model_lower:
        return (
            _sher_locker_google_base_url
            or _sher_locker_base_url
            or _DEFAULT_SHER_GOOGLE_BASE_URL
        )
    return (
        _sher_locker_openai_base_url
        or _sher_locker_base_url
        or _DEFAULT_SHER_OPENAI_BASE_URL
    )


class State(TypedDict):
    data_store: Dict[str, Any]
    image_store: List[Dict[str, Any]]
    html_store: List[Dict[str, Any]]
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
    visualize: bool


def _read_skill_description(skill_md_path: Path) -> Optional[str]:
    """
    Read `description:` from SKILL.md front matter.
    Returns None if the file or field is missing.
    """
    if not skill_md_path.exists():
        return None
    try:
        lines = skill_md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for raw in lines[:40]:
            line = raw.strip()
            if line.lower().startswith("description:"):
                value = line.split(":", 1)[1].strip()
                if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                    value = value[1:-1]
                return value or None
    except Exception:
        return None
    return None


def _tool_from_skill(func, name: str, skill_md_path: Path) -> StructuredTool:
    """Create a StructuredTool using SKILL.md description when available."""
    desc = _read_skill_description(skill_md_path)
    if desc:
        return StructuredTool.from_function(func=func, name=name, description=desc)
    return StructuredTool.from_function(func=func, name=name)


_SKILL_ROOT = Path(__file__).resolve().parent.parent / "skill"
tools = [
    _tool_from_skill(
        load_data,
        "load_data",
        _SKILL_ROOT / "load_data_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        load_geodata,
        "load_geodata",
        _SKILL_ROOT / "load_geodata_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        get_raster_path,
        "get_raster_path",
        _SKILL_ROOT / "get_raster_path_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        get_raster_description,
        "get_raster_description",
        _SKILL_ROOT / "get_raster_description_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        analyze_raster_overlap,
        "analyze_raster_overlap",
        _SKILL_ROOT / "analyze_raster_overlap_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        get_values_from_raster_with_geometries,
        "get_values_from_raster_with_geometries",
        _SKILL_ROOT / "get_values_from_raster_with_geometries_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        classify_raster_zones,
        "classify_raster_zones",
        _SKILL_ROOT / "classify_raster_zones_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        extract_raster_boundary,
        "extract_raster_boundary",
        _SKILL_ROOT / "extract_raster_boundary_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        convert_dem_to_slope,
        "convert_dem_to_slope",
        _SKILL_ROOT / "convert_dem_to_slope_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_raster_selection_area,
        "calculate_raster_selection_area",
        _SKILL_ROOT / "calculate_raster_selection_area_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        generate_aspect_map,
        "generate_aspect_map",
        _SKILL_ROOT / "generate_aspect_map_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        generate_profile_curvature_map,
        "generate_profile_curvature_map",
        _SKILL_ROOT / "generate_profile_curvature_map_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        rasterize_vector_to_match_raster,
        "rasterize_vector_to_match_raster",
        _SKILL_ROOT / "rasterize_vector_to_match_raster_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        merge_dataframes,
        "merge_dataframes",
        _SKILL_ROOT / "merge_dataframes_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        get_unique_values,
        "get_unique_values",
        _SKILL_ROOT / "get_unique_values_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        filter_categorical,
        "filter_categorical",
        _SKILL_ROOT / "filter_categorical_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        filter_numerical,
        "filter_numerical",
        _SKILL_ROOT / "filter_numerical_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_column_statistics,
        "calculate_column_statistics",
        _SKILL_ROOT / "calculate_column_statistics_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        make_choropleth_map,
        "make_choropleth_map",
        _SKILL_ROOT / "make_choropleth_map_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        filter_points_by_raster_values,
        "filter_points_by_raster_values",
        _SKILL_ROOT / "filter_points_by_raster_values_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        select_features_by_spatial_relationship,
        "select_features_by_spatial_relationship",
        _SKILL_ROOT / "select_features_by_spatial_relationship_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_line_lengths,
        "calculate_line_lengths",
        _SKILL_ROOT / "calculate_line_lengths_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        analyze_vector_overlap,
        "analyze_vector_overlap",
        _SKILL_ROOT / "analyze_vector_overlap_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        perform_vector_topology_operation,
        "perform_vector_topology_operation",
        _SKILL_ROOT / "perform_vector_topology_operation_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        random_sample_from_layer,
        "random_sample_from_layer",
        _SKILL_ROOT / "random_sample_from_layer_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_nearest_distances,
        "calculate_nearest_distances",
        _SKILL_ROOT / "calculate_nearest_distances_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_standard_directional_ellipse,
        "calculate_standard_directional_ellipse",
        _SKILL_ROOT / "calculate_standard_directional_ellipse_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_line_direction_rose,
        "calculate_line_direction_rose",
        _SKILL_ROOT / "calculate_line_direction_rose_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_columns,
        "calculate_columns",
        _SKILL_ROOT / "calculate_columns_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        scale_column_by_value,
        "scale_column_by_value",
        _SKILL_ROOT / "scale_column_by_value_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        create_statistical_chart,
        "create_statistical_chart",
        _SKILL_ROOT / "create_statistical_chart_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        reproject_vector_data,
        "reproject_vector_data",
        _SKILL_ROOT / "reproject_vector_data_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        convert_csv_to_shapefile,
        "convert_csv_to_shapefile",
        _SKILL_ROOT / "convert_csv_to_shapefile_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        make_heatmap,
        "make_heatmap",
        _SKILL_ROOT / "make_heatmap_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        create_point_kernel_density_map,
        "create_point_kernel_density_map",
        _SKILL_ROOT / "create_point_kernel_density_map_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        visualize_geographies,
        "visualize_geographies",
        _SKILL_ROOT / "visualize_geographies_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        plot_contour_lines,
        "plot_contour_lines",
        _SKILL_ROOT / "plot_contour_lines_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        generate_contours_display,
        "generate_contours_display",
        _SKILL_ROOT / "generate_contours_display_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        make_bivariate_map,
        "make_bivariate_map",
        _SKILL_ROOT / "make_bivariate_map_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        reject_task,
        "reject_task",
        _SKILL_ROOT / "reject_task_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        get_centroids,
        "get_centroids",
        _SKILL_ROOT / "get_centroids_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        create_dissolved_buffer,
        "create_dissolved_buffer",
        _SKILL_ROOT / "create_dissolved_buffer_statements" / "SKILL.md",
    ),
    _tool_from_skill(
        calculate_polygon_areas,
        "calculate_polygon_areas",
        _SKILL_ROOT / "calculate_polygon_areas_statements" / "SKILL.md",
    ),
]


def _display_image_from_base64(image_base64: str, description: Optional[str] = None) -> None:
    """Render a base64 PNG/JPEG image immediately in notebook environments."""
    if not image_base64:
        return
    if IPyImage is None or ipy_display is None:
        return
    try:
        raw = base64.b64decode(image_base64)
        ipy_display(IPyImage(data=raw))
        if description:
            print(f"[image] {description}")
    except Exception:
        # Rendering is best-effort and must not break task execution.
        return


def execute_task(
    task_text: str,
    temperature: float = 0,
    model: str = MODEL_GPT_4o,
    max_steps: int = 25,
    capture_history: bool = False,
):
    conversation_history = [] if capture_history else None
    solution_steps = []
    input_tokens = []
    output_tokens = []

    if model in [MODEL_CLAUDE, MODEL_CLAUDE_mini, MODEL_CLAUDE_ADV3, MODEL_CLAUDE_ADV4]:
        llm = ChatAnthropic(model=model, temperature=temperature)
    elif model in [MODEL_GPT_4o, MODEL_GPT_41, MODEL_GPT_mini]:
        llm = ChatOpenAI(model=model, temperature=temperature)
    elif model in [MODEL_O3, MODEL_O4]:
        llm = ChatOpenAI(model=model, temperature=None)
    elif model in [MODEL_GEMINI, MODEL_GEMINI_ADV]:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    elif model in [
        MODEL_SHER_LOCKER,
        MODEL_SHER_LOCKER_4o,
        MODEL_SHER_LOCKER_GEMINI_FLASH,
        MODEL_SHER_LOCKER_4mini,
    ]:
        if not _sher_locker_api_key:
            raise ValueError("SHER_LOCKER_API_KEY is not set in the environment.")
        base_url = _resolve_sher_locker_base_url(model)
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=_sher_locker_api_key,
            base_url=base_url,
        )
    else:
        raise ValueError("Model is outside the predetermined list")

    graph = create_react_agent(
        llm,
        tools=tools,
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

    def _extract_text_from_message(message_content) -> Optional[str]:
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, list):
            parts = []
            for item in message_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
            text = "\n".join(parts).strip()
            return text if text else None
        return None

    try:
        for s in graph.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            if hasattr(message, "usage_metadata") and message.usage_metadata:
                input_tokens.append(message.usage_metadata["input_tokens"])
                output_tokens.append(message.usage_metadata["output_tokens"])

            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    step = Step(function_name=tool_call["name"], arguments=tool_call["args"])
                    solution_steps.append(step)
            else:
                if getattr(message, "type", None) == "ai":
                    text = _extract_text_from_message(message.content)
                    if text:
                        final_message = text

            if capture_history:
                conversation_history.append({"type": message.type, "content": message.content})

            if "image_store" in s and s["image_store"]:
                for img_data in s["image_store"]:
                    _display_image_from_base64(
                        img_data.get("base64", ""),
                        img_data.get("description", "Visualization"),
                    )
                    if not capture_history:
                        continue
                    conversation_history.append(
                        {
                            "type": "image",
                            "content": img_data["base64"],
                            "description": img_data.get("description", "Visualization"),
                        }
                    )
                s["image_store"].clear()

            if capture_history and "html_store" in s and s["html_store"]:
                for html_item in s["html_store"]:
                    conversation_history.append(
                        {
                            "type": "interactive_map",
                            "content": html_item["html"],
                            "description": html_item.get("description", "Interactive Map"),
                        }
                    )
                s["html_store"].clear()

    except GraphRecursionError as e:
        print(f"Maximum recursion depth reached: {e}")

    solution = Solution(steps=solution_steps)
    return solution, input_tokens, output_tokens, conversation_history, final_message
