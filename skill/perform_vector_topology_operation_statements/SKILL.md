---
name: perform-vector-topology-operation
description: Perform vector topology operations (union, difference, intersection, symmetric_difference) between two GeoDataFrames, store the result in data_store, and save the vector output to the project root scratch folder.
---

# Perform Vector Topology Operation Skill

This skill wraps the canonical implementation in geobenchx.tools.

## Capabilities

- Run union, intersection, difference, and symmetric difference between two vector layers.
- Auto-align CRS before operation.
- Store result GeoDataFrame in shared state.
- Save result vector file to project root scratch folder with timestamped unique filename.

## How to Use

1. Ensure both input GeoDataFrames are loaded in shared state.
2. Call $(System.Collections.Hashtable.func) with operation and output names.
3. Reuse saved output path and data_store result in downstream analysis.

## Scripts

- $(System.Collections.Hashtable.py): Wrapper exposing geobenchx.tools.perform_vector_topology_operation.
