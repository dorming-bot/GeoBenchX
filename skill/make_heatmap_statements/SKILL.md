---
name: make-heatmap
description: The make_heatmap tool creates an interactive heatmap visualization from point data in a GeoDataFrame, with intensity based on values from a specified column. It returns a visualization where color intensity represents data density or magnitude, and can optionally store the HTML output. The tool supports customization of map appearance including basemap style, zoom level, and color radius.
---

# Make Heatmap Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.make_heatmap.
