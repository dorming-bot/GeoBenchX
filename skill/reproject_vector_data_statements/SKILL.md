---
name: reproject-vector-data
description: Reproject a vector GeoDataFrame to a target CRS and save the result to the project root scratch folder as a new vector file. Optionally stores the reprojected GeoDataFrame in data_store.
---

# Reproject Vector Data Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reproject vector layers to a requested CRS.
- Save the reprojected layer as a new vector file.
- Optionally keep the reprojected GeoDataFrame in shared state.

## How to Use

1. Ensure the source GeoDataFrame exists in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Use the saved file or stored GeoDataFrame in later steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.reproject_vector_data.
