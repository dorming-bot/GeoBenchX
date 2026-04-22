---
name: select-features-by-spatial-relationship
description: Select features from one GeoDataFrame based on multiple spatial relationships with another. Cannot process GeoDataFrame with over 150,000 features. Features that satisfy ANY of the specified predicates will be selected (OR logic). Automatically handles CRS differences by reprojecting features to match reference CRS. Returns flitered GeoDataFrame. Optionally, visualizes the filtered GeoDataFrame over reference GeoDataFrame. Also returns number of selected features, predicates used and name of the filtered dataframe.
---

# Select Features By Spatial Relationship Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Reuse the production-ready GeoBenchX tool behavior without duplicating logic.
- Keep outputs and state updates consistent with the original benchmark pipeline.

## How to Use

1. Ensure required inputs are present in shared state.
2. Call $(System.Collections.Hashtable.func) with the same arguments as in geobenchx.tools.
3. Consume returned summaries and state artifacts in downstream steps.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.select_features_by_spatial_relationship.
