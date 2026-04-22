# GeoBenchX Skills

This directory contains modular, self-documenting skills for geospatial task processing.

## Skill Structure

Each skill follows a standardized structure:

```
skill_name/
├── SKILL.md              # Skill documentation with YAML front-matter
├── skill_name.py         # Implementation
└── __init__.py           # Package initialization
```

## Available Skills

### Vector Processing Skills

- **load_geodata**: Load vector datasets from catalog into shared state
- **calculate_polygon_areas**: Measure polygon areas with automatic UTM projection
- **get_centroids**: Extract centroid points from polygon features
- **create_dissolved_buffer**: Create and dissolve buffer zones

### Raster Processing Skills

- **get_raster_path**: Resolve raster dataset paths from catalog
- **get_raster_description**: Extract raster metadata and statistics

## SKILL.md Format

Each SKILL.md must include YAML front-matter:

```markdown
---
name: skill-name
description: Brief description of what the skill does
---

# Skill Title

Detailed documentation...
```

## Usage

Skills are automatically registered as LangChain StructuredTools in `skill_bench/agent.py`:

```python
from skill_test.skill.load_geodata import load_geodata
from skill_test.skill.calculate_polygon_areas import calculate_polygon_areas
```
