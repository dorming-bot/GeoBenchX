---
name: filter-categorical
description: Filters DataFrame/GeoDataFrame by categorical values in specified columns. Can be used to select specific countries, subregions, continents from respective columns. To get better results, can be used after the values for filtering are compared with spellings in the selected column using get_unique_values_tool. Returns filtered DataFrame/GeoDataFrame, filters applied, details on columns and rows of the new dataframe.
---

# Filter Categorical Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.filter_categorical.
