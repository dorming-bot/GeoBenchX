---
name: generate-contours-display
description: Creates contour lines from raster data using GDAL's ContourGenerate algorithm. It returns a GeoDataFrame containing line geometries that represent areas of equal value with associated attributes. To get values for the function arguments, call get_raster_description_tool first. Tool supports customization of contour intervals, value ranges. Note, that common nodata values could be -99999, -9999, -32768, -999, -3.4e38  and assign function arguments accordingly. The tool also produces a shapefile that can be optionally saved, optionally can visualize background raster and generated contour lines and customize styling.
---

# Generate Contours Display Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.generate_contours_display.
