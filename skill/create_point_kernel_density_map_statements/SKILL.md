---
name: create-point-kernel-density-map
description: Generates a kernel density surface from point data, auto-selecting a sensible bandwidth when not provided, storing the grid metadata, and returning a basemap-backed visualization that highlights hotspot areas.
---

# Create Point Kernel Density Map Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.create_point_kernel_density_map.
