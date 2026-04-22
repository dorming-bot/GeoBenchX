---
name: visualize-geographies
description: Displays multiple GeoDataFrames as map layers with custom styling. It returns a visualization with different geometries (points, lines, polygons) rendered in distinct colors over a basemap, with optional legends and titles. The tool handles coordinate system transformations automatically and supports various basemap styles.
---

# Visualize Geographies Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.visualize_geographies.
