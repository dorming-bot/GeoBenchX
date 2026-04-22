---
name: filter-points-by-raster-values
description: Sample raster values at point locations and filter points based on threshold conditions. Adds these values to a specified column in the GeoDataFrame. Returns filtered GeoDataFrame. Optionally, visualizes the filtered points overlaid on the raster. Also returns text summary with filtering statistics (total points, filtered points, name of the column cotaining sampled values).
---

# Filter Points By Raster Values Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.filter_points_by_raster_values.
