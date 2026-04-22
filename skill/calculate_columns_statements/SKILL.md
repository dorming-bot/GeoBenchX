---
name: calculate-columns
description: Performs mathematical operations ("multiply", "divide", "add", "subtract") between columns of two DataFrames/GeoDataFrames or columns of the same DataFrame/GeoDataFrame. Returns DataFrame/GeoDataFrame with a column containing the operation result and message about operation, resulting dataframe and column.
---

# Calculate Columns Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.calculate_columns.
