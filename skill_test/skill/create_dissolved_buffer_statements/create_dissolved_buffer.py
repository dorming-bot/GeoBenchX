"""Create dissolved buffers with preview map and optional file exports."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import exceptions as pyproj_exceptions
from langgraph.prebuilt import InjectedState

from skill_test.skill_bench.utils import get_dataframe_info
from skill_test.skill_bench.constants import SKILL_SCRATCH_PATH

SCRATCH_PATH = Path(os.getenv("SKILL_SCRATCHPATH", str(SKILL_SCRATCH_PATH)))

BASEMAP_PROVIDERS = {
    "OpenStreetMap": ctx.providers.OpenStreetMap.Mapnik,
    "Carto Positron": ctx.providers.CartoDB.Positron,
    "Carto Dark": ctx.providers.CartoDB.DarkMatter,
}


def create_dissolved_buffer(
    geodataframe_name: str,
    buffer_size_meters: float,
    output_geodataframe_name: str,
    state: Annotated[Dict[str, Any], InjectedState],
    dissolve_by_attribute: Optional[str] = None,
    output_file_path: Optional[str] = None,
    overwrite_existing: bool = True,
    basemap_style: Literal["OpenStreetMap", "Carto Positron", "Carto Dark"] = "Carto Positron",
    plot_title: Optional[str] = None,
    **_: Any,
) -> str:
    """Mirror the behavior of geobenchx.tools.create_dissolved_buffer."""
    if "data_store" not in state:
        state["data_store"] = {}
    if "image_store" not in state:
        state["image_store"] = []

    geodataframe = state["data_store"].get(geodataframe_name)
    if geodataframe is None:
        return f"Error: GeoDataFrame '{geodataframe_name}' not found in data store"
    if geodataframe.empty:
        return "Error: Source GeoDataFrame is empty"
    if geodataframe.crs is None:
        return "Error: Source GeoDataFrame has no CRS. Please assign one before buffering."

    try:
        metric_crs = geodataframe.estimate_utm_crs()
    except pyproj_exceptions.CRSError:
        metric_crs = None

    metric_crs_code = metric_crs.to_string() if metric_crs is not None else "EPSG:3857"
    geodataframe_metric = geodataframe.to_crs(metric_crs_code)
    buffered = geodataframe_metric.copy()
    buffered.geometry = buffered.geometry.buffer(buffer_size_meters)

    if dissolve_by_attribute:
        if dissolve_by_attribute not in buffered.columns:
            return f"Error: Column '{dissolve_by_attribute}' not found in GeoDataFrame"
        dissolved = buffered.dissolve(by=dissolve_by_attribute).reset_index()
    else:
        dissolved = gpd.GeoDataFrame(
            {"buffer_id": ["merged_buffer"], "geometry": [buffered.geometry.unary_union]},
            crs=buffered.crs,
        )

    dissolved_original_crs = dissolved.to_crs(geodataframe.crs)
    state["data_store"][output_geodataframe_name] = dissolved_original_crs

    if output_file_path is None:
        SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
        output_file_path = SCRATCH_PATH / f"{output_geodataframe_name}.shp"
    vector_path = Path(output_file_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    driver_lookup = {
        ".shp": "ESRI Shapefile",
        ".gpkg": "GPKG",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
    }
    ext = vector_path.suffix.lower()
    if ext not in driver_lookup:
        return f"Error: Unsupported output file extension '{ext}'. Use .shp, .gpkg, or .geojson."

    if vector_path.exists():
        if not overwrite_existing:
            return f"Error: Output file '{vector_path.as_posix()}' exists. Enable overwrite_existing to replace it."
        if ext == ".shp":
            for shp_ext in [".shp", ".shx", ".dbf", ".cpg", ".prj", ".sbx", ".sbn"]:
                candidate = vector_path.with_suffix(shp_ext)
                if candidate.exists():
                    candidate.unlink()
        else:
            vector_path.unlink()

    dissolved_original_crs.to_file(vector_path.as_posix(), driver=driver_lookup[ext])

    provider = BASEMAP_PROVIDERS.get(basemap_style, ctx.providers.CartoDB.Positron)
    buffer_plot = dissolved_original_crs.to_crs("EPSG:3857")
    source_plot = geodataframe.to_crs("EPSG:3857")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    buffer_plot.plot(
        ax=ax,
        color="#2ca25f",
        edgecolor="#1b7837",
        linewidth=1.2,
        alpha=0.5,
        label=f"{buffer_size_meters:.0f} m Buffer",
    )
    source_plot.plot(ax=ax, color="#045a8d", linewidth=0.8, alpha=0.8, label="Original features")
    basemap_warning: Optional[str] = None
    try:
        ctx.add_basemap(ax, source=provider, crs="EPSG:3857")
    except Exception as exc:
        basemap_warning = f"Basemap not added due to connection issue: {exc}"
        ax.set_facecolor("#f7f7f7")
    ax.set_axis_off()
    map_title = plot_title or f"{buffer_size_meters:.0f} m Dissolved Buffer"
    ax.set_title(map_title)
    legend_handles = [
        Patch(
            facecolor="#2ca25f",
            edgecolor="#1b7837",
            linewidth=1.2,
            alpha=0.5,
            label=f"{buffer_size_meters:.0f} m Buffer",
        ),
        Line2D([0], [0], color="#045a8d", linewidth=0.8, label="Original features"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    state["image_store"].append(
        {
            "type": "map",
            "description": f"{map_title} for {geodataframe_name}",
            "base64": img_base64,
        }
    )

    info_string, _ = get_dataframe_info(dissolved_original_crs)
    dissolve_note = (
        f"Dissolved by column '{dissolve_by_attribute}'."
        if dissolve_by_attribute
        else "All buffered features merged into a single geometry."
    )

    summary = [
        f"Created {buffer_size_meters:.0f}m buffer from '{geodataframe_name}' "
        f"and stored dissolved result as '{output_geodataframe_name}'.",
        dissolve_note,
        f"Input features: {len(geodataframe)}, output features: {len(dissolved_original_crs)}.",
        f"GeoDataFrame details:\n{info_string}",
        f"Vector file saved to: {vector_path.as_posix()}",
    ]
    if basemap_warning:
        summary.append(f"Basemap warning: {basemap_warning}")

    return "\n".join(summary)
