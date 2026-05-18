---
name: create-3d-dem-visualization
description: Create an interactive 3D DEM surface visualization, optionally overlay vector GeoDataFrames as 3D points or lines draped on the DEM, and save the result as an HTML file in the project root scratch folder.
---

# Create 3D DEM Visualization Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Render a DEM raster as an interactive Plotly 3D surface.
- Downsample large DEMs for responsive browser viewing.
- Overlay optional vector GeoDataFrames as 3D markers or draped lines.
- Save the final interactive result as an HTML file in the project root `scratch` folder.
- Store output path, display grid size, elevation range, CRS, and overlay metadata in shared state.

## How to Use

1. Provide `dem_raster_path`, `output_variable_name`, and shared `state`.
2. Optionally provide `vector_geodataframe_names` as a list or JSON-string list.
3. Tune `max_grid_size`, `z_exaggeration`, `surface_colormap`, and `vector_z_offset` if needed.
4. Use the returned HTML path to inspect the interactive 3D visualization.

## Scripts

- `create_3d_dem_visualization.py`: Wrapper exposing `geobenchx.tools.create_3d_dem_visualization`.
