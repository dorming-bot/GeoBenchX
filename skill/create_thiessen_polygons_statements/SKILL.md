---
name: create-thiessen-polygons
description: Create Thiessen (Voronoi) polygons from a point GeoDataFrame, optionally clip by a boundary GeoDataFrame, store the result in data_store, and save vector output to the project root scratch folder.
---

# Create Thiessen Polygons Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Build Thiessen (Voronoi) polygons from point features.
- Support optional clipping by a boundary GeoDataFrame.
- Store generated polygons in shared state for downstream analysis.
- Save output vector file under project scratch folder.

## How to Use

1. Provide point GeoDataFrame name in data_store.
2. Provide output GeoDataFrame name.
3. Optionally provide clip_geodataframe_name and output_file_name.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.create_thiessen_polygons.
