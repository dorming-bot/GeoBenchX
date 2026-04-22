---
name: merge-dataframes
description: Merge statistical and geospatial dataframes using specified key columns. The resulting merged dataframe will preserve all rows from geodataset, matching with dataset where possible and filling with NaN where no match exists. Can work also on 2 DataFrame or a2 GeoDataFrames. Returns the DataFrame and its description, inlcuding DataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.
---

# Merge Dataframes Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.merge_dataframes.
