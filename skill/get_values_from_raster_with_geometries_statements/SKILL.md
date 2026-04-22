---
name: get-values-from-raster-with-geometries
description: Mask a raster using vector geometries from a GeoDataFrame and calculate statistics for the masked area of the raster. The tool returns statistics (total value in masked areas, min, max, mean, standard deviation, count) and stores the cropped raster data in the specified variable. Optionally displays a visualization of masked areas.
---

# Raster Values by Geometry Skill

This skill extracts raster values using geometry masks from a GeoDataFrame and computes summary statistics.

## Capabilities

- Mask raster pixels by polygon/geometry extents.
- Compute aggregate metrics for masked raster cells.
- Store result dictionaries and optional visualization artifacts.

## How to Use

1. Ensure raster path and source GeoDataFrame are available.
2. Call `get_values_from_raster_with_geometries` with raster path, geometry key, and output variable.
3. Use stored summaries for thresholds, filtering, or reporting.

## Inputs

- `raster_path` - Raster file path/URI.
- `geodataframe_name` - Key of vector geometries in `state["data_store"]`.
- `output_variable_name` - Key for storing extracted raster statistics.
- `state` - Shared LangGraph state.
- Optional toggles for plotting and save behavior.

## Outputs

- Statistics package stored in `state["data_store"][output_variable_name]`.
- Optional plot preview in `state["image_store"]`.
- A textual summary message of extraction results.

## Scripts

- `get_values_from_raster_with_geometries.py`: Wrapper exposing `geobenchx.tools.get_values_from_raster_with_geometries`.
