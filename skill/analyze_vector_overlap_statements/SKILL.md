---
name: analyze-vector-overlap
description: Intersect two GeoDataFrames and store the resulting overlap layer for downstream measurements or visualization, including point-vs-line/polygon overlap, and save the overlap vector output to the project root scratch folder.
---

# Analyze Vector Overlap Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.analyze_vector_overlap.
