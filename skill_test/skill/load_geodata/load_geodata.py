"""Load vector datasets listed in the SKILL catalog into the shared state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import geopandas as gpd
from langgraph.prebuilt import InjectedState

from skill_test.skill_bench.utils import get_dataframe_info

CATALOG_PATH = Path(__file__).resolve().parents[2] / "dataset" / "data_catalogs_snapshot.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GEODATA_PATH = Path(
    os.getenv("GEODATAPATH", REPO_ROOT / "data" / "Data" / "GeoData")
)

with CATALOG_PATH.open("r", encoding="utf-8") as fp:
    _catalog_snapshot = json.load(fp)

GEO_DATASETS: Dict[str, str] = _catalog_snapshot.get("GEO_CATALOG", {})
_LOWER_TO_LABEL: Dict[str, str] = {name.lower(): name for name in GEO_DATASETS}
_FILENAME_TO_LABEL: Dict[str, str] = {
    Path(file_name).name.lower(): label for label, file_name in GEO_DATASETS.items()
}


def available_datasets() -> List[str]:
    """Return the catalog names users can request."""
    return sorted(GEO_DATASETS.keys())


def _resolve_dataset_path(file_name: str) -> Path:
    root = Path(DEFAULT_GEODATA_PATH)
    return (root / file_name).resolve()


def _infer_dataset(geodataset: str) -> Tuple[str, Path]:
    """
    Determine the canonical dataset label and resolved path for the provided
    input. Accepts catalog display names, raw file names, or absolute paths.
    """
    normalized = geodataset.strip()
    lower = normalized.lower()

    # Direct catalog label (with case-insensitive support)
    if lower in _LOWER_TO_LABEL:
        label = _LOWER_TO_LABEL[lower]
        return label, _resolve_dataset_path(GEO_DATASETS[label])

    # File name references (e.g., "Chengdu_Surface.shp")
    if lower in _FILENAME_TO_LABEL:
        label = _FILENAME_TO_LABEL[lower]
        return label, _resolve_dataset_path(GEO_DATASETS[label])

    # Absolute or relative path (including stems without .shp)
    candidate = Path(normalized)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".shp")
    if candidate.exists():
        return normalized, candidate.resolve()

    candidate = (DEFAULT_GEODATA_PATH / normalized).with_suffix(candidate.suffix)
    if candidate.exists():
        return normalized, candidate.resolve()

    raise FileNotFoundError(
        f"Dataset '{geodataset}' was not found. Provide one of the catalog names "
        f"({', '.join(sorted(GEO_DATASETS))}) or an existing file path."
    )


def load_geodata(
    geodataset: str,
    output_geodataframe_name: str,
    state: Annotated[Dict[str, Any], InjectedState],
    **_: Any,
) -> str:
    """
    Read a GeoDataset into GeoPandas and store it in ``state['data_store']``.
    """
    if not output_geodataframe_name:
        return "Error: output_geodataframe_name must be provided."

    try:
        canonical_label, dataset_path = _infer_dataset(geodataset)
    except FileNotFoundError as exc:
        return f"Error: {exc}"

    geodataframe = gpd.read_file(dataset_path)

    data_store = state.setdefault("data_store", {})
    data_store[output_geodataframe_name] = geodataframe

    # Register aliases so downstream tool calls can reference either catalog
    # labels or raw filenames without re-loading the dataset.
    alias_candidates = {
        canonical_label,
        Path(canonical_label).name,
        Path(dataset_path).name,
        Path(dataset_path).stem,
    }
    alias_candidates.update({alias.lower() for alias in alias_candidates if isinstance(alias, str)})
    for alias in alias_candidates:
        if alias and alias not in data_store:
            data_store[alias] = geodataframe

    info_string, non_empty = get_dataframe_info(geodataframe)
    return (
        f"Loaded geodata '{canonical_label}' from '{dataset_path}'.\n"
        f"Stored as '{output_geodataframe_name}'.\n"
        f"GeoDataFrame description:\n{info_string}\n"
        f"Non-empty values summary:\n{non_empty}"
    )
