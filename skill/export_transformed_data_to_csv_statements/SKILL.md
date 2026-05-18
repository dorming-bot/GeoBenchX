---
name: export-transformed-data-to-csv
description: Transform DataFrame/GeoDataFrame content by requirement (geometry coordinates, pairwise point distances, point-to-line distances, or spatial aggregation) and save the transformed result as a CSV file in the project root scratch folder; supports direct aggregation on an already intersected source layer and strict selected_columns export.
---

# Export Transformed Data To CSV Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Convert vector geometries into coordinate records and export CSV.
- Compute pairwise distances between point features and export CSV.
- Compute point-to-line (or point-to-geometry) distances and export CSV.
- Aggregate numeric values spatially across intersecting regions and export CSV.
- Aggregate directly on a single already-intersected layer when no secondary layer is provided.
- Export only requested columns via selected_columns (for strict outputs like index + P).
- Optionally keep the transformed table in shared state for downstream tools.

## How to Use

1. Provide source dataframe name from data_store and output CSV name.
2. Select transformation mode.
3. For distance modes, provide secondary dataframe as needed.
4. For spatial aggregation, provide value_column and optionally secondary_dataframe_name.
5. Use selected_columns when output must contain exact columns only.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.export_transformed_data_to_csv.
