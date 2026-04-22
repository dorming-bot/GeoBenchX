---
name: filter-numerical
description: Filters DataFrame/GeoDataFrame using numerical conditions via query method (e.g. "col1 > 25 and col2 < 100"). To identify most suitable values for the filter, calculate_column_statistics_tool can be used before filtering. Returns filtered DataFrame/GeoDataFrame, filters applied, details on columns and rows of the new dataframe.
---

# Filter Numerical Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.filter_numerical.
