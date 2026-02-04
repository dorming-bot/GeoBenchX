name: create-dissolved-buffer
description: Create buffer zones in meters, dissolve overlaps, export vector files, and capture preview maps.
---

# Dissolved Buffer Generation Skill

This skill mirrors GIS desktop workflows for building buffers with optional attribute dissolves, high-quality map
renders, and on-disk exports ready for sharing or reuse.

## Capabilities

- Reproject input features into a metric CRS (auto-estimated UTM or EPSG:3857) before buffering.
- Merge overlapping buffers globally or per attribute value to maintain categorical boundaries.
- Persist GeoDataFrames inside the LangGraph state, save shapefile/GPKG/GeoJSON outputs, and push a base64 map preview.
- Provide human-readable summaries describing buffer size, feature counts, and save locations.

## How to Use

1. **Prepare Source Layer**: Load polygons/lines into `state["data_store"]` via `load_geodata` or other ingestion steps.
2. **Configure Buffer**: Choose distance, dissolve behavior, optional output path, and basemap style.
3. **Execute Skill**: Call the wrapper script; review the returned summary and inspect `state["image_store"]` for preview imagery.

## Inputs

- `geodataframe_name` – Key of the source GeoDataFrame.
- `buffer_size_meters` – Numeric buffer radius in meters.
- `output_geodataframe_name` – Storage key for the dissolved results in state.
- `state` – Shared LangGraph state.
- Optional: `dissolve_by_attribute`, `output_file_path`, `overwrite_existing`, `basemap_style`, `plot_title`.

## Outputs

- `state["data_store"][output_geodataframe_name]` containing dissolved buffers in the original CRS.
- Optional file written to disk (shp/gpkg/geojson) plus base64 preview appended to `state["image_store"]`.
- Text summary highlighting dissolve rules, feature totals, and saved file paths.

## Scripts

- `create_dissolved_buffer.py`: Entry point that invokes `geobenchx.tools.create_dissolved_buffer`.

## Notes

- Attribute dissolves validate that the column exists before processing.
- When overwriting shapefiles, all companion files (.shx, .dbf, etc.) are cleaned up for consistency.
