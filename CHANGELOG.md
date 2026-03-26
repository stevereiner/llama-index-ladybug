# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `docs/api/README.md` — hand-written API reference for `LadybugGraphStore` and `LadybugPropertyGraphStore` covering all public constructors, methods, and properties
- `docs/notebooks/property_graph_ladybug.ipynb` — property graph example notebook using `PropertyGraphIndex` with structured schema, vector index, and combined graph+vector retrieval
- `docs/notebooks/LadybugGraphDemo.ipynb` — legacy knowledge graph example notebook using `KnowledgeGraphIndex` with pyvis visualization
- `docs/notebooks/ladybuggraph_draw.html` — pre-rendered interactive pyvis graph visualization
- `docs/notebooks/README.md` — overview of both notebooks and when to use each

### Changed
- Renamed PyPI package from `llama-index-graph-stores-ladybug` to `llama-index-ladybug` in `pyproject.toml`, API docs, and both notebooks
- Bumped initial package version to `0.15.2` to align with the minimum supported `real-ladybug` version
- Added `nbstripout>=0.9.0` to dev dependencies and `.pre-commit-config.yaml` with an `nbstripout` hook so notebook outputs are always stripped before commits, regardless of whether Cursor has rerun them

### Fixed
- Upgraded `real-ladybug` dependency from `0.12.0` to `>=0.15.2,<0.16` to resolve a native access violation crash when calling `CREATE_VECTOR_INDEX` on Windows with Python 3.13
- Fixed deprecation warning in `base.py` by replacing separate `prepare()` + `execute()` calls with a single `execute()` call as required by the updated `real-ladybug` API
