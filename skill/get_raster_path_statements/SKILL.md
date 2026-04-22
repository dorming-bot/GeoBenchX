name: get-raster-path
description: Resolve normalized paths or ZIP URIs for raster datasets registered in the catalog.
---

# Raster Path Resolution Skill

This lightweight skill standardizes access to raster datasets. It ensures every raster request returns a platform-safe
path (or `zip://` URI) ready for Rasterio, GDAL, or other raster workflows.

## Capabilities

- Validate raster names against the curated `RASTER_CATALOG`.
- Produce POSIX-style paths rooted at `GEODATAPATH` for consistent cross-platform usage.
- Detect zipped rasters and build `zip://{folder}/{archive}!/{tif}` URIs automatically.

## How to Use

1. **Select Dataset**: Identify the raster label (e.g., “USA population 2020, people, resolution 1 km”).
2. **Invoke Skill**: Call the wrapper script with the dataset name and shared state.
3. **Consume Path**: Use the returned path inside downstream skills like `plot_contour_lines`, masking, or raster sampling.

## Inputs

- `rasterdataset` – Key defined in `RASTER_CATALOG`.
- `state` – Shared LangGraph state object (maintains consistent API parity with other tools).

## Outputs

- String message containing the fully resolved raster path or ZIP URI.

## Scripts

- `get_raster_path.py`: Local implementation that reads the SKILL raster catalog and emits normalized file or ZIP URIs.

## Notes

- The skill assumes a `.tif` with the same base filename exists inside a `.zip` archive.
- Keep `GEODATAPATH` updated to point at the folder housing both GeoTIFFs and compressed datasets.
