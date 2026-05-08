---
name: convert-csv-to-shapefile
description: Convert a CSV file with coordinate columns into a point shapefile and save it to the project root scratch folder. Supports explicit or auto-detected X/Y columns and optional data_store output.
---

# Convert CSV To Shapefile Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Read CSV records and detect or use specified XY coordinate columns.
- Convert rows to point geometries.
- Save output as a timestamp-suffixed .shp file in the project root scratch folder to keep filenames unique.
- Optionally store the generated GeoDataFrame in shared state.

## How to Use

1. Provide a CSV file path (or a loaded DataFrame name in data_store) and output shapefile name.
2. Optionally provide x/y columns and source CRS.
3. Use the saved shapefile path (and optional state output) in downstream tools.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.convert_csv_to_shapefile.
