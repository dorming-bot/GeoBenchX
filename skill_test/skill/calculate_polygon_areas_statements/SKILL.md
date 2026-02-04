name: calculate-polygon-areas
description: Measure polygon footprints in square kilometers using auto-selected UTM projections and store detailed state metadata.
---

# Polygon Area Measurement Skill

This skill delivers QGIS-style polygon area calculations. It reprojects features into an appropriate UTM (or an
equal-area fallback), appends per-feature area columns, and captures total statistics for downstream analytics.

## Capabilities

- Automatically determine the most suitable UTM zone, with Albers Equal Area fallback for edge cases.
- Append configurable area columns (square kilometers) to the GeoDataFrame.
- Store projection info, totals, and enriched GeoDataFrames inside the shared LangGraph state.
- Provide concise textual summaries for chat responses or logging.

## How to Use

1. **Load Polygons**: Use `load_geodata` or another ingestion step to place polygon features inside `state["data_store"]`.
2. **Call Skill**: Provide the dataset key, output variable name, and optional `area_column_name`.
3. **Consume Results**: Reference `state["data_store"][output_variable_name]` for totals or reuse the projected GeoDataFrame
   in overlap, buffering, or visualization tasks.

## Inputs

- `geodataframe_name` – Identifier of the polygon layer already stored in state.
- `output_variable_name` – Name for the results package containing GeoDataFrame + statistics.
- `state` – LangGraph state object shared by the agent.
- `area_column_name` *(optional)* – Custom column name for per-feature area values.

## Outputs

- `state["data_store"][output_variable_name]` with fields such as projected GeoDataFrame, EPSG code, UTM zone, totals, and labels.
- Text summary describing CRS decisions and total area for prompt injection or logs.

## Scripts

- `calculate_polygon_areas.py`: Standalone implementation that selects UTM zones, computes per-feature areas, and writes summaries to the SKILL state.

## Notes

- Errors are surfaced when the GeoDataFrame is missing, empty, or lacks a CRS, ensuring fail-fast agent flows.
- Square-kilometer units simplify comparisons with demographic or land-use indicators used elsewhere in GeoBenchX.
