---
name: scale-column-by-value
description: Performs a basic mathematical operation (multiply, divide, add, or subtract) between a column's values and a specified numeric value. It returns a new DataFrame/GeoDataFrame with the original data plus a new column containing the calculation results as well as and message about operation, resulting dataframe and column.
---

# Scale Column By Value Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.scale_column_by_value.
