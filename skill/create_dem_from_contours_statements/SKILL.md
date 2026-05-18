---
name: create-dem-from-contours
description: Create a DEM elevation raster from a contour vector GeoDataFrame by sampling contour elevations and interpolating to a regular grid, saving the GeoTIFF in the project root scratch folder.
---

# Create DEM From Contours Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Read contour line or point geometries from `data_store`.
- Sample elevation values from the contour attribute field and interpolate a DEM grid.
- Save the output raster as a GeoTIFF in the project root `scratch` folder.
- Store output path, statistics, and processing metadata in shared state.
- Capture a preview image for inspection.

## How to Use

1. Provide the contour GeoDataFrame name, elevation column, output variable name, and shared state.
2. Optionally provide `pixel_size`, `width`, `height`, `interpolation_method`, `target_crs`, and output path settings.
3. Use the returned summary and stored raster path in downstream DEM or terrain analysis steps.

## Scripts

- `create_dem_from_contours.py`: Wrapper exposing `geobenchx.tools.create_dem_from_contours`.
