---
name: calculate-column-statistics
description: Calculate summary statistics (count (total, valid, and missing values), min/max range, mean, standard deviation) for a numerical column in a DataFrame/GeoDataFrame. Optionally calculates quantiles (25th, 50th/median, 75th percentiles). Can calculate additional custom quantiles if specified. Returns a formatted summary of the statistics. The tool can be used before applying numerical filters to calculate the optimal filtering thresholds.
---

# Calculate Column Statistics Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.calculate_column_statistics.
