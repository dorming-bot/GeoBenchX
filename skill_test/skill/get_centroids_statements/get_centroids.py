"""Generate centroid features and preview maps from polygon GeoDataFrames."""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from langgraph.prebuilt import InjectedState
from pyproj import exceptions as pyproj_exceptions

from skill_test.skill_bench.utils import get_dataframe_info
from skill_test.skill_bench.constants import SKILL_SCRATCH_PATH

SCRATCH_PATH = Path(os.getenv("SKILL_SCRATCHPATH", str(SKILL_SCRATCH_PATH)))

BASEMAP_PROVIDERS = {
    "OpenStreetMap": ctx.providers.OpenStreetMap.Mapnik,
    "Carto Positron": ctx.providers.CartoDB.Positron,
    "Carto Dark": ctx.providers.CartoDB.DarkMatter,
}


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_centroids(
    geodataframe_name: str,
    output_geodataframe_name: str,
    state: Annotated[Dict[str, Any], InjectedState],
    create_centroid_for_each_part: bool = False,
    output_shapefile_path: Optional[str] = None,
    output_tif_path: Optional[str] = None,
    title: Optional[str] = None,
    basemap_style: Literal["OpenStreetMap", "Carto Positron", "Carto Dark"] = "Carto Positron",
    overwrite_existing: bool = True,
    output_variable_name: Optional[str] = None,
    **_: Any,
) -> str:
    """
    Compute centroids for polygons, optionally explode multipart geometries,
    and store both the centroids GeoDataFrame and preview artifacts.
    """
    if "data_store" not in state:
        state["data_store"] = {}
    if "image_store" not in state:
        state["image_store"] = []

    source_gdf = state["data_store"].get(geodataframe_name)
    if source_gdf is None:
        return f"Error: GeoDataFrame '{geodataframe_name}' not found in state."
    if source_gdf.empty:
        return f"Error: GeoDataFrame '{geodataframe_name}' has no features."
    if source_gdf.crs is None:
        return "Error: GeoDataFrame must have a CRS before calculating centroids."

    if create_centroid_for_each_part:
        try:
            polygons = source_gdf.explode(index_parts=False).reset_index(drop=True)
        except TypeError:
            polygons = source_gdf.explode().reset_index(drop=True)
    else:
        polygons = source_gdf.copy()

    try:
        metric_crs = polygons.estimate_utm_crs()
    except pyproj_exceptions.CRSError:
        metric_crs = None
    metric_crs_code = metric_crs.to_string() if metric_crs is not None else "EPSG:3857"

    polygons_metric = polygons.to_crs(metric_crs_code)
    centroids_metric = polygons_metric.geometry.centroid

    centroids_gdf = polygons_metric.copy()
    centroids_gdf.geometry = centroids_metric
    centroids_gdf = centroids_gdf.to_crs(source_gdf.crs)
    state["data_store"][output_geodataframe_name] = centroids_gdf

    scratch_root = SCRATCH_PATH
    scratch_root.mkdir(parents=True, exist_ok=True)

    shapefile_path = Path(output_shapefile_path) if output_shapefile_path else scratch_root / f"{output_geodataframe_name}.shp"
    if shapefile_path.suffix.lower() != ".shp":
        return "Error: output_shapefile_path must end with .shp to match the expected centroid export format."
    _ensure_directory(shapefile_path)
    if shapefile_path.exists():
        if not overwrite_existing:
            return f"Error: Shapefile '{shapefile_path}' exists. Enable overwrite to replace."
        for ext in [".shp", ".shx", ".dbf", ".cpg", ".prj", ".sbn", ".sbx"]:
            candidate = shapefile_path.with_suffix(ext)
            if candidate.exists():
                candidate.unlink()
    centroids_gdf.to_file(shapefile_path.as_posix())

    tif_path = Path(output_tif_path) if output_tif_path else scratch_root / f"{output_geodataframe_name}_map.tif"
    _ensure_directory(tif_path)
    if tif_path.exists():
        if not overwrite_existing:
            return f"Error: Preview TIFF '{tif_path}' exists. Enable overwrite to replace."
        tif_path.unlink()

    provider = BASEMAP_PROVIDERS.get(basemap_style, ctx.providers.CartoDB.Positron)
    polygons_web = source_gdf.to_crs(epsg=3857)
    centroids_web = centroids_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    polygons_web.plot(ax=ax, color="#4292c6", edgecolor="#08519c", linewidth=0.8, alpha=0.25, label="Source polygons")
    centroids_web.plot(ax=ax, color="#d7301f", markersize=40, marker="*", label="Centroids")
    bounds = polygons_web.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    basemap_warning: Optional[str] = None
    try:
        ctx.add_basemap(ax, source=provider, crs="EPSG:3857")
    except Exception as exc:  # Network outages should not fail the entire tool.
        basemap_warning = f"Basemap not added due to connection issue: {exc}"
        ax.set_facecolor("#f7f7f7")
    ax.set_axis_off()
    map_title = title or "Feature centroids"
    ax.set_title(map_title)
    if len(centroids_web) > 1:
        ax.legend(loc="lower left")

    fig.savefig(tif_path.as_posix(), format="tif", dpi=200, bbox_inches="tight", facecolor="white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    image_store = state.setdefault("image_store", [])
    image_store.append(
        {
            "type": "map",
            "description": f"{map_title} for {geodataframe_name}",
            "base64": img_base64,
        }
    )

    if output_variable_name:
        summary = {
            "source_geodataframe": geodataframe_name,
            "output_geodataframe": output_geodataframe_name,
            "centroid_count": int(len(centroids_gdf)),
            "create_centroid_for_each_part": create_centroid_for_each_part,
            "shapefile_path": shapefile_path.as_posix(),
            "map_tif_path": tif_path.as_posix(),
            "crs": str(centroids_gdf.crs),
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bounds": centroids_gdf.total_bounds.tolist(),
        }
        state["data_store"][output_variable_name] = summary

    info_string, _ = get_dataframe_info(centroids_gdf)
    summary_lines = [
        f"Created {len(centroids_gdf)} centroid(s) from '{geodataframe_name}'.",
        f"- Per-part centroids: {'enabled' if create_centroid_for_each_part else 'disabled'}",
        f"- Output GeoDataFrame: '{output_geodataframe_name}'",
        f"- Shapefile saved to: '{shapefile_path.as_posix()}'",
        f"- Map TIFF saved to: '{tif_path.as_posix()}'",
        f"GeoDataFrame details:\n{info_string}",
    ]
    if basemap_warning:
        summary_lines.insert(-1, f"- Basemap warning: {basemap_warning}")
    if output_variable_name:
        summary_lines.insert(-1, f"- Summary stored under: '{output_variable_name}'")

    return "\n".join(summary_lines)
