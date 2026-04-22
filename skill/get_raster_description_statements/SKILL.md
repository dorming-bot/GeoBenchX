---
name: get-raster-description
description: Inspect raster metadata and per-band statistics, optionally storing the results in SKILL state.
---

# Raster Description Skill

This skill opens a raster path (GeoTIFF or ZIP-backed dataset) and reports driver,
dimensions, CRS, NoData values, and descriptive statistics for each band. It mirrors
the diagnostic output a GIS analyst would inspect before processing rasters.

## Capabilities

- Read raster metadata such as driver, dimensions, CRS, pixel dtype, and NoData value.
- Compute min/max/mean/std/valid pixel counts per band, masking NoData elements.
- Store the metadata/statistics dictionary inside `state["data_store"]` when an
  `output_variable_name` is supplied.
- Return a formatted string that can be added to the agent's chat transcript or logs.

## How to Use

1. Ensure the raster file is accessible (local GeoTIFF or ZIP path).
2. Call `get_raster_description` with the raster path and shared `state`.
3. Optionally provide `output_variable_name` to retain the metadata for later steps.

## Inputs

- `raster_path` – Absolute or relative path to the raster file.
- `state` – LangGraph state object carrying the shared data stores.
- `output_variable_name` *(optional)* – Key to store the metadata/statistics dict.

## Outputs

- Text summary that lists metadata and per-band statistics.
- Optional entry in `state["data_store"]` named by `output_variable_name`.

## Script

- `get_raster_description.py`: Standalone implementation that opens the raster with Rasterio and computes the summary without relying on `geobenchx`.

## Notes

- The skill fails fast if a raster cannot be opened or lacks valid pixels.
- Statistics ignore NoData pixels by default, ensuring meaningful summaries for partially masked rasters.
