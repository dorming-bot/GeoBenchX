---
name: create-statistical-chart
description: Create histogram, bar, or pie charts from DataFrame/GeoDataFrame columns or raster pixel values (QGIS Histogram-style). Stores chart metadata in data_store and chart image preview in image_store.
---

# Create Statistical Chart Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Generate histogram, bar chart, and pie chart outputs for raster and vector/statistical workflows.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state (or provide raster path for histogram).
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and image artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.create_statistical_chart.
