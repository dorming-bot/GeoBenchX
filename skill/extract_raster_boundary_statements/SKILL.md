---
name: extract-raster-boundary
description: Convert the valid pixels of a raster into a simplified polygon boundary, optionally saving a lightweight vector file for further spatial analysis.
---

# Raster Boundary Extraction Skill

This skill converts valid raster areas into simplified boundary geometry outputs.

## Capabilities

- Detect valid raster footprints and polygonize boundaries.
- Simplify/extract lightweight boundary vectors from raster masks.
- Save outputs for subsequent overlay, buffering, or reporting tasks.

## How to Use

1. Resolve the target raster path.
2. Call `extract_raster_boundary` with output parameters and shared state.
3. Reuse resulting boundary geometries in vector workflows.

## Inputs

- `raster_path` - Input raster path/URI.
- `output_variable_name` / output geometry name - State key for boundary outputs.
- `state` - Shared LangGraph state.
- Optional simplification/export parameters.

## Outputs

- Boundary vector data stored in `state["data_store"]` under the requested key.
- Optional saved vector file and summary metadata.
- A textual extraction summary.

## Scripts

- `extract_raster_boundary.py`: Wrapper exposing `geobenchx.tools.extract_raster_boundary`.
