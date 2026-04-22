---
name: get-unique-values
description: Gets unique values from a specified column in a DataFrame/GeoDataFrame. Should be used to clarify the spelling of names of ojects like countries, regions, subregions, continents, etc. before using the filtering tools to avoid missing objects due to using differeing spelling or convention. Returns list of unique values from a specified column in a DataFrame/GeoDataFrame.
---

# Get Unique Values Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.get_unique_values.
