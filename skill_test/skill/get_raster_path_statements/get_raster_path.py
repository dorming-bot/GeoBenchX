"""Resolve raster catalog entries without depending on geobenchx."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List

from langgraph.prebuilt import InjectedState

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RASTER_PATH = Path(os.getenv("RASTERPATH", REPO_ROOT / "data" / "Data" / "GeoData"))

RASTER_CATALOG = {
    "Accumulated snow cover season 2023-2024, USA, inches": "sfav2_CONUS_2023093012_to_2024093012_processed.tif",
    "Accumulated snow cover season 2024-2025, USA, inches": "sfav2_CONUS_2024093012_to_2025052012_processed.tif",
    "Tibetan Plato South Asia flood extent, August 2018": "DFO_4665_From_20180802_to_20180810.tif",
    "Bangladesh population, 2018, people, resolution 1 km": "bgd_pd_2018_1km_UNadj.tif",
    "USA population 2020, people, resolution 1 km": "usa_ppp_2020_1km_Aggregated_UNadj.tif",
    "Chile population, 2020, people, resolution 1 km": "chl_ppp_2020_1km_Aggregated_UNadj.tif",
    "Angola population, 2020, people, resolution 1 km": "ago_ppp_2020_1km_Aggregated_UNadj.tif",
    "Peru, Bolivia, Argentina, Chile flood, February 2018": "DFO_4569_From_20180201_to_20180221.tif",
    "Peru population, 2018, 1 km resolution": "per_ppp_2018_1km_Aggregated_UNadj.tif",
    "Brazil population, 2018, 1 km resolution": "bra_ppp_2018_1km_Aggregated_UNadj.tif",
    "Algeria population density per 1 km 2020, 1 km resolution": "dza_pd_2020_1km_UNadj.tif",
    "Wei River Basin Topographic Slope Classification Dataset": "WeiheBasin.tif",
    "XinxiangCity Rainstorm and Flooding Dataset": "Xinxiang City Rainstorm and Flooding Dataset.tif",
    "Guangming District DEM (Shenzhen)": "guangming.tif",
    "Guangming District Slope (Shenzhen)": "guangmingSlope.tif",
    "DEM data of the Tangxun Lake Experimental Area in Hubei": "Tangxun_Lake_DEM.tif",
    "Piura region of Peru": "piura_dem.tif",
}


def available_rasters() -> List[str]:
    """Return available raster catalog entries."""
    return sorted(RASTER_CATALOG.keys())


def _resolve_dataset_path(file_name: str) -> Path:
    root = DEFAULT_RASTER_PATH
    path = (root / file_name).resolve()
    return path


def get_raster_path(
    rasterdataset: str,
    state: Annotated[Dict[str, Any], InjectedState],
) -> str:
    """Construct a usable path (GeoTIFF or ZIP) for the requested raster."""
    if rasterdataset not in RASTER_CATALOG:
        return (
            f"Error: Raster dataset '{rasterdataset}' not found. "
            f"Available datasets: {', '.join(available_rasters())}"
        )

    file_name = RASTER_CATALOG[rasterdataset]
    dataset_path = _resolve_dataset_path(file_name)

    if not dataset_path.exists():
        return f"Error: Raster file '{dataset_path.as_posix()}' does not exist."

    if dataset_path.suffix.lower() == ".zip":
        tif_name = dataset_path.with_suffix(".tif").name
        raster_path = f"zip://{dataset_path.as_posix()}!/{tif_name}"
    else:
        raster_path = dataset_path.as_posix()

    return f"Constructed path to raster dataset '{rasterdataset}': {raster_path}"
