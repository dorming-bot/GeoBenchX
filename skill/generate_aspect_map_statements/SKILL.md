---
name: generate-aspect-map
description: Creates an aspect raster (0-360 degrees) from a DEM using GDAL, saves a timestamped GeoTIFF, and renders a Spectral singleband pseudocolor preview with summary statistics.
---

# Generate Aspect Map Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.generate_aspect_map.
