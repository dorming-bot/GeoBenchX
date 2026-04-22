---
name: load-geodata
description: Resolve cataloged vector datasets and place the resulting GeoDataFrame plus aliases inside shared SKILL state.
---

# Load Geodata Skill

This skill loads any vector dataset listed in the SKILL catalog (or referenced by file name / path),
places the GeoDataFrame into `state["data_store"]`, and registers helpful aliases so downstream
skills can reuse the data without hitting disk again.

## Capabilities

- Accept catalog labels, raw shapefile names, or explicit paths and resolve them to absolute files.
- Read the dataset via GeoPandas and store the GeoDataFrame under `output_geodataframe_name`.
- Register multiple aliases (catalog label, file name, stem, lowercase versions) that all point to the
  same GeoDataFrame within the shared state.
- Return a concise description of the dataset, including `GeoDataFrame.info()` text and non-empty stats,
  which can be streamed back to the agent transcript.

## How to Use

1. Inspect the dataset catalog in the prompt to identify the canonical dataset name.
2. Call `load_geodata` with that name (or a matching file/path) plus the state handle and desired
   `output_geodataframe_name`.
3. Read the summary string to confirm row counts, CRS, and column completeness before calling other skills.

## Inputs

- `geodataset` – Catalog label, file name, or existing path to a vector dataset.
- `output_geodataframe_name` – Key for storing the loaded GeoDataFrame.
- `state` – LangGraph state object (used to persist GeoDataFrames).

## Outputs

- GeoDataFrame stored under `output_geodataframe_name` and aliased names inside `state["data_store"]`.
- Text summary describing the dataset path, storage key, and `GeoDataFrame.info()` details.

## Scripts

- `load_geodata.py`: Resolves dataset names/paths, loads them with GeoPandas, and updates state metadata.

## Notes

- Dataset lookup relies on `dataset/data_catalogs_snapshot.json` and defaults to `data/Data/GeoData`
  unless `GEODATAPATH` overrides it.
- Errors are human-readable (missing dataset names, nonexistent files, missing output name).
- No CRS transformations are performed; downstream skills should verify CRS before spatial operations.
