---
name: classify-raster-zones
description: Classify or reclassify raster values based on range definitions (min <= value <= max), similar to QGIS Reclassify by Table. Produces a new raster with specified dtype, stores class statistics, and visualizes the resulting zones with optional custom colors/labels.
---

# Raster Zone Classification Skill

This skill reclassifies raster values based on user-defined rules and generates zone-level outputs.

## Capabilities

- Apply manual range-based class definitions to raster cells.
- Produce classified raster outputs and class statistics.
- Optionally create visualization previews for classified zones.

## How to Use

1. Resolve an input raster path.
2. Provide classification ranges and target output variable names.
3. Run `classify_raster_zones` and inspect stored zone summaries.

## Inputs

- `raster_path` - Source raster path.
- `classification_rules` (or equivalent rule arguments) - Class boundaries/values.
- `output_variable_name` - State key for results.
- `state` - Shared LangGraph state.
- Optional plotting/saving parameters.

## Outputs

- Classification result package stored in `state["data_store"][output_variable_name]`.
- Optional preview image(s) in `state["image_store"]`.
- A textual classification summary.

## Scripts

- `classify_raster_zones.py`: Wrapper exposing `geobenchx.tools.classify_raster_zones`.
