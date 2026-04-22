---
name: calculate-line-direction-rose
description: Analyzes the orientations of line features (roads, rivers, etc.) by computing a rose diagram with length-weighted bins, returning a summary table, metadata, and an illustrative plot.
---

# Calculate Line Direction Rose Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.calculate_line_direction_rose.
