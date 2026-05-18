---
name: create-dissolved-buffer
description: Create QGIS-style buffer zones in meters, dissolve overlaps, export vector files, and capture preview maps with configurable segments, cap style, join style, miter limit, and processing CRS.
---

# Dissolved Buffer Generation Skill

This skill mirrors GIS desktop workflows for building buffers with optional attribute dissolves, high-quality map
renders, and on-disk exports ready for sharing or reuse.

## Capabilities

- Match QGIS Buffer defaults more closely: segments=5, round caps, round joins, and mitre_limit=2.0.
- Use the source CRS when it is already projected in meters; otherwise reproject into an auto-estimated metric CRS.
- Accept an explicit `buffer_crs` when the processing CRS must match a QGIS run exactly.
- Merge overlapping buffers globally or per attribute value to maintain categorical boundaries.
- Persist GeoDataFrames inside the LangGraph state, save shapefile/GPKG/GeoJSON outputs, and push a base64 map preview.
- Provide human-readable summaries describing buffer size, feature counts, and save locations.

## How to Use

1. **Prepare Source Layer**: Load polygons/lines into `state["data_store"]` via `load_geodata` or other ingestion steps.
2. **Configure Buffer**: Choose distance, dissolve behavior, optional output path, basemap style, and QGIS-style buffer parameters if needed.
3. **Execute Skill**: Call the wrapper script; review the returned summary and inspect `state["image_store"]` for preview imagery.

## Inputs

- `geodataframe_name` – Key of the source GeoDataFrame.
- `buffer_size_meters` – Numeric buffer radius in meters.
- `output_geodataframe_name` – Storage key for the dissolved results in state.
- `state` – Shared LangGraph state.
- Optional: `dissolve_by_attribute`, `output_file_path`, `overwrite_existing`, `buffer_crs`, `segments`, `cap_style`, `join_style`, `mitre_limit`, `basemap_style`, `plot_title`.

## Outputs

- `state["data_store"][output_geodataframe_name]` containing dissolved buffers in the original CRS.
- Optional file written to disk (shp/gpkg/geojson) plus base64 preview appended to `state["image_store"]`.
- Text summary highlighting dissolve rules, processing CRS, buffer parameters, feature totals, and saved file paths.

## Scripts

- `create_dissolved_buffer.py`: Entry point that invokes `geobenchx.tools.create_dissolved_buffer`.

## Notes

- Attribute dissolves validate that the column exists before processing.
- If QGIS was run in a specific projected CRS, pass the same CRS through `buffer_crs` for the closest match.
- When overwriting shapefiles, all companion files (.shx, .dbf, etc.) are cleaned up for consistency.
