---
name: convert-dem-to-slope
description: Converts a DEM raster into a slope raster in degrees (similar to QGIS Slope), saves a timestamped GeoTIFF into the scratch folder, stores summary statistics, and adds a preview image.
---

# Convert Dem To Slope Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.convert_dem_to_slope.
