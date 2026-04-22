---
name: calculate-line-lengths
description: Calculates the lengths of line features in a GeoDataFrame in kilometers, using appropriate UTM projections for accurate distance measurements. Returns total lengh of line features in km, GeoDataFrame in UTM projection with column storing features length in m, and a descritive message of the results.
---

# Calculate Line Lengths Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.calculate_line_lengths.
