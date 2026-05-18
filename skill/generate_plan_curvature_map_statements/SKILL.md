---
name: generate-plan-curvature-map
description: Creates a plan curvature raster from a DEM, saves a timestamped GeoTIFF in the project root scratch folder, captures a diverging preview map, and stores summary statistics.
---

# Generate Plan Curvature Map Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Compute DEM-derived plan curvature using finite-difference first- and second-order derivatives.
- Save the generated curvature raster as a GeoTIFF in the project root scratch folder.
- Store output raster path, statistics, source DEM path, timestamp, and method metadata in shared state.
- Capture a diverging preview map in the image store.

## How to Use

1. Provide `dem_raster_path`, `output_variable_name`, and shared `state`.
2. Optionally provide `output_raster_path`, `overwrite_existing`, `append_timestamp_to_output`, `plot_title`, and `colormap`.
3. Consume the returned summary, output raster path, and state artifacts in downstream steps.

## Scripts

- `generate_plan_curvature_map.py`: Wrapper exposing `geobenchx.tools.generate_plan_curvature_map`.
