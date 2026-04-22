---
name: rasterize-vector-to-match-raster
description: Rasterizes a GeoDataFrame (such as buffered geometries) so it matches the extent, CRS, and resolution of an existing raster, producing a GeoTIFF suitable for subsequent raster algebra operations.
---

# Rasterize Vector To Match Raster Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.rasterize_vector_to_match_raster.
