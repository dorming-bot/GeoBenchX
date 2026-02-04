"""Compute raster metadata and per-band statistics without geobench dependencies."""

from __future__ import annotations

import os
from typing import Annotated, Any, Dict, Optional

import numpy as np
import rasterio
from langgraph.prebuilt import InjectedState


def get_raster_description(
    raster_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    output_variable_name: Optional[str] = None,
) -> str:
    """Summarize raster metadata, statistics, and optionally store them in state."""
    if not raster_path:
        return "Error: raster_path must be provided."

    try:
        with rasterio.open(raster_path) as src:
            metadata = {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "transform": src.transform.to_gdal(),
                "nodata": src.nodata,
                "dtype": str(src.dtypes[0]),
            }

            stats = []
            for band in range(1, src.count + 1):
                band_data = src.read(band)
                if src.nodata is not None:
                    mask = band_data != src.nodata
                else:
                    mask = np.ones_like(band_data, dtype=bool)

                valid = band_data[mask]
                if valid.size > 0:
                    band_stats = {
                        "band": band,
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                        "mean": float(np.mean(valid)),
                        "std": float(np.std(valid)),
                        "nan_count": int(np.sum(~np.isfinite(valid))),
                        "valid_pixels": int(np.sum(mask)),
                    }
                else:
                    band_stats = {
                        "band": band,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                        "nan_count": None,
                        "valid_pixels": 0,
                    }
                stats.append(band_stats)

    except FileNotFoundError:
        return f"Error: Raster file '{raster_path}' not found."
    except rasterio.errors.RasterioIOError as exc:
        return f"Error: Unable to read raster '{raster_path}': {exc}"

    result = {"metadata": metadata, "statistics": stats}
    if output_variable_name:
        if "data_store" not in state:
            state["data_store"] = {}
        state["data_store"][output_variable_name] = result

    description = [
        f"Raster Dataset: {os.path.basename(raster_path)}",
        "\nMetadata:",
        f"- Dimensions: {metadata['width']} x {metadata['height']} pixels",
        f"- Number of bands: {metadata['count']}",
        f"- Coordinate System: {metadata['crs']}",
        f"- Data type: {metadata['dtype']}",
        f"- NoData value: {metadata['nodata']}",
        "\nBand Statistics:",
    ]

    for band_stats in stats:
        description.extend(
            [
                f"\nBand {band_stats['band']}:",
                f"- Valid pixels: {band_stats['valid_pixels']:,}",
                f"- Range: {band_stats['min']:.6g} to {band_stats['max']:.6g}"
                if band_stats["min"] is not None
                else "- Range: No valid data",
                f"- Mean: {band_stats['mean']:.6g}"
                if band_stats["mean"] is not None
                else "- Mean: No valid data",
                f"- Standard deviation: {band_stats['std']:.6g}"
                if band_stats["std"] is not None
                else "- Standard deviation: No valid data",
                f"- NaN count: {band_stats['nan_count']:,}"
                if band_stats["nan_count"] is not None
                else "- NaN count: No valid data",
            ]
        )

    return "\n".join(description)
