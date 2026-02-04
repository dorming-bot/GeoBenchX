---
name: get-centroids
description: Extract centroids from polygon features, export shapefiles and preview maps, and log metadata to state.
---

# Centroid Extraction Skill

This skill produces centroid geometries for polygon datasets, mirroring GIS desktop workflows with optional
per-part centroid creation, shapefile/TIFF exports, and detailed state tracking.

## Capabilities

- Generate centroid points for each polygon or each multipart segment.
- Store centroid GeoDataFrames inside `state["data_store"]`, export shapefiles, and save preview TIFFs plus base64 PNGs.
- Record metadata (counts, bounds, CRS, timestamps) under a caller-provided state key.
- Provide concise completion summaries for downstream prompts.

## How to Use

1. **Load Polygons**: Ensure the source GeoDataFrame is present in state and has a valid CRS.
2. **Configure Outputs**: Decide if multipart polygons should be exploded, and optionally set custom file paths/titles.
3. **Run Skill**: Call the wrapper script. Review the returned summary and inspect saved shapefiles or TIFF previews.

## Inputs

- `geodataframe_name` – Key of the polygon dataset in `state["data_store"]`.
- `output_geodataframe_name` – Key for storing centroid results.
- `state` – Shared LangGraph state object.
- Optional: `create_centroid_for_each_part`, `output_shapefile_path`, `output_tif_path`, `title`,
  `basemap_style`, `overwrite_existing`, `output_variable_name`.

## Outputs

- Centroid GeoDataFrame stored under `output_geodataframe_name`.
- Optional metadata dict saved to `output_variable_name`.
- Shapefile + TIFF preview (default paths in SCRATCH) plus base64 PNG appended to `state["image_store"]`.
- Text summary detailing centroid counts and export paths.

## Scripts

- `get_centroids.py`: Wrapper exposing `geobenchx.tools.get_centroids` for agent workflows.

## Notes

- Shapefile outputs enforce `.shp` extensions and clean up sidecar files when overwriting.
- Basemap styles support OpenStreetMap, Carto Positron, and Carto Dark through Contextily providers.
