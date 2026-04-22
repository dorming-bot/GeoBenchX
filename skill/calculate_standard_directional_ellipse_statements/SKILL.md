---
name: calculate-standard-directional-ellipse
description: Computes the Standard Deviational Ellipse (directional distribution) for a set of points, summarizing major/minor axes, orientation, and providing an ellipse GeoDataFrame plus visualization.
---

# Calculate Standard Directional Ellipse Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.calculate_standard_directional_ellipse.
