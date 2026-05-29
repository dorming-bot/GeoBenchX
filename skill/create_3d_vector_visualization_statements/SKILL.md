---
name: create-3d-vector-visualization
description: Create an interactive 3D visualization from vector GeoDataFrame geometries with Z coordinates or a specified height column, save the HTML output in the project root scratch folder, and store metadata in shared state.
---

# Create 3D Vector Visualization Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Render point/line/polygon vector geometries in Plotly 3D.
- Use geometry Z values directly when present.
- Fallback to an explicit numeric height column when geometry has no Z.
- Save the interactive result as an HTML file under the project root `scratch` folder.
- Store output path, CRS, height range, and rendering metadata in shared state.

## How to Use

1. Provide `geodataframe_name`, `output_variable_name`, and shared `state`.
2. Optionally provide `z_column` if source geometries are 2D but have a height attribute.
3. Optionally tune `z_exaggeration`, `point_size`, and `line_width`.
4. Inspect the returned HTML path for the interactive 3D view.

## Scripts

- `create_3d_vector_visualization.py`: Wrapper exposing `geobenchx.tools.create_3d_vector_visualization`.
