---
name: analyze-raster-overlap
description: Analyze overlap between two rasters (raster 1 and raster 2) and calculate statistics (total value, min, max, mean, standard deviation, count) for overlapping pixels from raster 2. Optionally displays a visualization of masked areas.
---

# Raster Overlap Analysis Skill

This skill compares two rasters in shared extent/CRS space and summarizes overlap-region metrics.

## Capabilities

- Align and mask two rasters to their common overlap area.
- Compute overlap statistics and optional difference summaries.
- Optionally save derived overlap outputs and visualization artifacts.

## How to Use

1. Resolve raster paths (typically via `get_raster_path`).
2. Call `analyze_raster_overlap` with both raster paths and output variable name.
3. Reuse stored summary metrics in later arithmetic or reporting steps.

## Inputs

- `raster1_path`, `raster2_path` - File paths or URIs to rasters.
- `output_variable_name` - State key for storing analysis results.
- `state` - Shared LangGraph state.
- Optional parameters for resampling, plotting, saving overlap raster, and differences.

## Outputs

- Overlap statistics stored in `state["data_store"][output_variable_name]`.
- Optional image previews appended to `state["image_store"]`.
- A textual status summary of overlap analysis.

## Scripts

- `analyze_raster_overlap.py`: Wrapper exposing `geobenchx.tools.analyze_raster_overlap`.
