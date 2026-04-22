"""Calculate polygon areas using a locally managed implementation."""

from __future__ import annotations

from typing import Annotated, Any, Dict

import numpy as np
from langgraph.prebuilt import InjectedState


def calculate_polygon_areas(
    geodataframe_name: str,
    output_variable_name: str,
    state: Annotated[Dict[str, Any], InjectedState],
    area_column_name: str = "area_sq_km",
) -> str:
    """Measure polygon areas in square kilometers and store results in state."""
    if "data_store" not in state:
        state["data_store"] = {}

    geodataframe = state["data_store"].get(geodataframe_name)
    if geodataframe is None:
        return f"Error: GeoDataFrame '{geodataframe_name}' not found"
    if geodataframe.empty:
        return f"Error: GeoDataFrame '{geodataframe_name}' has no features"
    if geodataframe.geometry.is_empty.all():
        return f"Error: GeoDataFrame '{geodataframe_name}' contains empty geometries"
    if geodataframe.crs is None:
        return "Error: GeoDataFrame must have a CRS before calculating areas"

    centroid = geodataframe.geometry.unary_union.centroid
    utm_zone = int(np.floor((centroid.x + 180) / 6) + 1)
    hemisphere = "N" if centroid.y >= 0 else "S"
    epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

    fallback_crs = "ESRI:102025"  # Asia North Albers Equal Area
    projection_label = f"UTM zone {utm_zone}{hemisphere} (EPSG:{epsg})"

    try:
        projected = geodataframe.to_crs(f"EPSG:{epsg}")
    except Exception:
        projected = geodataframe.to_crs(fallback_crs)
        projection_label = f"Fallback CRS {fallback_crs}"

    projected[area_column_name] = projected.geometry.area / 1_000_000
    total_area_sq_km = float(projected[area_column_name].sum())

    result = {
        "gdf_with_areas": projected,
        "total_area_sq_km": total_area_sq_km,
        "area_column": area_column_name,
        "utm_zone": utm_zone,
        "epsg": epsg,
        "projection_used": projection_label,
    }
    state["data_store"][output_variable_name] = result

    return (
        f"Calculated polygon areas in '{geodataframe_name}' using {projection_label}.\n"
        f"Total area: {total_area_sq_km:,.2f} square kilometers\n"
        f"Results stored in '{output_variable_name}' with per-feature areas in column '{area_column_name}'"
    )
