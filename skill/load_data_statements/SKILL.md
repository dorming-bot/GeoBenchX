---
name: load-data
description: Loads statistical data from the DATA_CATALOG into a Pandas DataFrame. Returns the DataFrame and its description, inlcuding DataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.
---

# Tabular Data Loading Skill

This skill loads catalog-based statistical datasets into the shared state as a Pandas DataFrame.

## Capabilities

- Resolve dataset names using the canonical `DATA_CATALOG` in `geobenchx.tools`.
- Load CSV/ZIP-backed tabular datasets and store them in `state["data_store"]`.
- Return structural summaries for downstream tool planning.

## How to Use

1. Choose a dataset label from the GeoBenchX data catalog.
2. Call `load_data` with `dataset`, `output_dataframe_name`, and `state`.
3. Reuse the resulting DataFrame key in later filtering/merge/statistics tools.

## Inputs

- `dataset` - Dataset label defined in the canonical GeoBenchX catalog.
- `output_dataframe_name` - State key used to store the loaded DataFrame.
- `state` - Shared LangGraph state object.

## Outputs

- A loaded DataFrame stored in `state["data_store"][output_dataframe_name]`.
- A textual summary of schema and non-empty values.

## Scripts

- `load_data.py`: Wrapper exposing `geobenchx.tools.load_data`.
