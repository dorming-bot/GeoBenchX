---
name: plot-contour-lines
description: creates topographic-style contour lines from raster data at specified intervals. It returns a GeoDataFrame containing line geometries representing areas of equal value (e.g., elevation, temperature), with proper coordinate transformation for mapping. Also returns message with contour interval, number of contour features, range of values, crs, name of teh resulting GeoDataFrame. The tool can also generate an optional visualization of the contours with customizable styling.
---

# Plot Contour Lines Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.plot_contour_lines.
