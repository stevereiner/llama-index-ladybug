# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [0.3.4] - 2026-07-21

### Changed
- Bumped minimum `ladybug` from `>=0.16.1` to `>=0.18.2`. As of 0.18.x, vector indexing (`CREATE_VECTOR_INDEX`) lives in a loadable VECTOR extension rather than core — see the extension-load fix below. The extension is downloaded at runtime via `INSTALL vector;` (needs network the first time, then cached under `~/.lbdb/extension/`); there is no `ladybug[vector]` pip extra. See the README "Vector index extension" section for offline / Docker / corporate-proxy pre-bake instructions.

### Fixed
- **`get_triplets()` raised `KeyError: '_label'`** (`ladybug_property_graph.py`) — the loop used bare bracket access (`record["e"]["_label"]`, `["_id"]`, `record["r"]["_src"]`/`["_dst"]`) assuming lowercase internal metadata keys, but `structured_query` returns them uppercase (`_LABEL`/`_ID`/`_SRC`/`_DST`, as `get_rel_map` and `vector_query` already handle). Now reads either case via `.get()`, skips malformed or `Chunk` rows instead of raising, filters both `_id`/`_ID` and `_label`/`_LABEL` out of node properties, and defaults a missing entity label to `"Entity"`.
- **Vector extension load failures were silently swallowed** (`ladybug_property_graph.py`) — the `INSTALL vector; LOAD vector;` setup ran as one statement wrapped in `except RuntimeError: pass`, so if `INSTALL` errored (already-installed or unreachable) `LOAD` was skipped, and any failure surfaced later as a confusing `Catalog exception: function CREATE_VECTOR_INDEX is not defined` (hit on Ladybug 0.18.x, where vector indexing moved into the loadable VECTOR extension). `INSTALL` and `LOAD` now run separately (a benign `INSTALL` error no longer skips `LOAD`), and a real `LOAD` failure logs a warning with remediation instead of being discarded.
- **`LOAD vector` failed on Windows with `error 126` from a plain `cmd`/PowerShell venv** (`ladybug_property_graph.py`) — the downloaded VECTOR extension imports `libssl-3-x64.dll` / `libcrypto-3-x64.dll` by plain name, but the ladybug wheel ships those OpenSSL DLLs delvewheel-mangled in `ladybug.libs`, so the engine's native loader couldn't find them (`IO exception: Failed to load library … libvector.lbug_extension … The specified module could not be found`) unless an unmangled copy happened to be on `PATH` (e.g. Git's `mingw64\bin`). The store now stages unmangled copies into `~/.lbdb/dll_shim` and adds it to the DLL search path on init, so `LOAD` works regardless of how Python was launched. (Upstream ladybug packaging gap; this is a defensive shim.)

## [0.3.3] - 2026-05-12

### Fixed
- **`vector_query` crash on empty database** — calling `vector_query` before any nodes have been
  upserted raised a `Binder` or `chunk_embedding_index` error because Ladybug cannot create a vector
  index on an empty table. The public `vector_query` method now catches those specific errors and
  returns `([], [])`, allowing callers to handle an empty result normally. The implementation body
  was extracted to `_vector_query_impl` so subclasses can override just the logic without losing the
  guard.

## [0.3.2] - 2026-05-07

### Changed
- Updated `ladybug` dependency from `>=0.15.3,<0.16` to `>=0.16.1` to track the new 0.16.x release line; bumped package version to `0.3.2`

## [0.3.1] - 2026-04-12

### Changed
- Renamed dependency from `real-ladybug` to `ladybug` across the entire repo (PyPI package was renamed); updated `pyproject.toml`, `uv.lock`, all source files (`base.py`, `utils.py`, `ladybug_property_graph.py`), tests, notebooks, `README.md`, and `docs/api/README.md`
- Bumped minimum `llama-index-core` from `>=0.13.0` to `>=0.14.20` in `pyproject.toml` and `README.md` requirements to align with current flexible-graphrag usage
- Bumped Python minimum from `3.9+` to `3.10+` in `pyproject.toml` and `README.md`
- Bumped package version from `0.3.0` to `0.3.1`
- `uv.lock` regenerated from adsharma's py3.14 base lock (PR #1) with updated `ladybug 0.15.3` (renamed from `real-ladybug`), `llama-index-core 0.14.20`, and `llama-index-graph-stores-ladybug 0.3.1` entries
- Added `import ladybug as lb` to both import blocks in `docs/api/README.md` so `lb.Database` / `lb.Connection` constructor references are self-contained

### Fixed
- `ChunkNode(metadata=...)` deprecation in `test_bugs.py` — replaced with `ChunkNode(properties=...)` for llama-index-core 0.14.x compatibility

## [2026-04-10]

### Changed
- Merged PR #1 from adsharma/dependencies — updated `pyproject.toml` and `uv.lock` to add Python 3.14 (`cp314`) wheel support for `real-ladybug 0.15.3`

## [0.3.0] - 2026-04-03

### Changed
- Package version set to `0.3.0`

### Fixed
- **Entity nodes not deleted on incremental delete** — `delete(properties={'doc_id': ...})` matched nothing because entity tables had no `ref_doc_id` column; added `ref_doc_id STRING` to entity DDL, write it in upsert, and match on it directly in `delete()` with a MENTIONS-traversal fallback

## [2026-04-02]

### Fixed
- **`get_rel_map` NoneType crash** — added `pivot_response = pivot_response or []` null-guard when `structured_query` returns `None`
- **Chunk embedding not stored on fresh database** — upsert now uses DELETE → DROP index → CREATE with embedding inline; previous SET-after-rebuild raised `Cannot set property vec`
- **`value_sanitize()` drops all MENTIONS at >= 128 rows** — `llama_index.core` silently returns `None` for lists of 128+ items; all MENTIONS queries and entity traversal now use `connection.execute` directly with `LIMIT 512`
- **`DETACH DELETE` wipes MENTIONS from other documents** — chunk pre-delete now matches on both `id` and `ref_doc_id`; skips CREATE on UUID collision with a different document
- **`Binder exception` when entity label matches a relation table name** — `safe_label()` now checks existing relation tables and appends `_TYPE` on collision
- **`NameError: node_tables` in `upsert_relations`** — moved `node_tables` assignment to top of method
- **Relation table name collides with node table** — `safe_rel_label()` appends `_REL` when name matches an existing node table

### Added
- Regression tests for the `value_sanitize` 128-row threshold (`tests/test_bugs.py`)

## [2026-04-01]

### Fixed
- **Reserved-keyword relation table collision** — `safe_label()` renamed both node and rel labels to `_TYPE`, causing a native crash; added `safe_rel_label()` using `_REL` suffix for relation tables
- **Schema-defined relations silently not stored** — `create_relation_tables()` only registered the first `FROM/TO` pair per label; additional pairs now added via `ALTER TABLE ADD IF NOT EXISTS`; `upsert_relations` switched to typed `MATCH` + `ensure_rel_table()` before each `MERGE`

### Changed
- Bumped minimum `real-ladybug` dependency to `>=0.15.3,<0.16` and package version to `0.15.3`

## [2026-03-31]

### Added
- **Structured schema — strict mode** (`strict_schema=True`) — entity/relation types not present in the ontology are rejected at ingest time
- **Structured schema — non-strict mode** (`strict_schema=False`) — off-schema LLM-extracted types are stored as additional node/relation table types alongside the schema-defined ones
- **Unstructured mode** (`has_structured_schema=False`) — no schema enforced; all entity/relation types stored freely
- **Vector index** (`use_vector_index=True`, default) — HNSW vector index on `Chunk.embedding`; `_create_vector_index` / `_ensure_vector_indexes` / `refresh_vector_index` manage lifecycle; index is dropped before chunk upsert and rebuilt after
- **`vector_query()`** — vector similarity search on `Chunk` nodes; expands results to entity nodes via `MENTIONS` so `VectorContextRetriever` scoring works correctly; falls back to raw chunk nodes when no `MENTIONS` links exist; distance converted to cosine similarity
- **`avector_query()`** — async wrapper that dispatches `vector_query()` to Ladybug's `AsyncConnection` thread-pool executor so FastAPI's event loop is not blocked
- **`get_rel_map()`** — single-hop entity traversal using `MATCH (e)-[rel]->(other)`; `MENTIONS` and `Chunk` targets filtered in Python (Ladybug Cypher does not reliably support `NOT (n:Label)` predicates); propagates `triplet_source_id` from `vector_query` entity nodes to triplet source nodes for `add_source_text()` lookup
- **`get_llama_nodes()` / `aget_llama_nodes()`** — fetches `Chunk` nodes by id directly from Ladybug (in-memory LlamaIndex docstore is empty after restart); returns `TextNode` objects with `file_name`, `file_path`, `ref_doc_id` metadata so `add_source_text()` can attach source text and filename to search results
- **`_checkpoint()`** — flushes WAL to disk after writes; called after `upsert_relations` and `vector_query`
- **`_initialize_embedding_dimension()` / `_detect_embedding_dimension()`** — auto-detects embedding dimension from embed model via a test embedding; falls back to explicit `embedding_dimension` parameter
- **Incremental updates** — add, modify, and delete via Alfresco STOMP events; `document_state` records created correctly across all schema modes
- **`utils.ensure_rel_table()`** — idempotent `CREATE REL TABLE IF NOT EXISTS` + `ALTER TABLE ADD IF NOT EXISTS FROM … TO …`; used by `upsert_relations` for both off-schema and schema-defined relations
- **`utils.ensure_links_pair()`** — adds a new `FROM/TO` pair to the `LINKS` catch-all table
- **`utils.remove_empty_values()`** — strips `None` / empty-string values from property dicts before passing to LlamaIndex node constructors

### Fixed
- **Incremental upsert with vector index** — second `upsert_nodes` call raised `RuntimeError: Cannot set property vec in table embeddings`; fixed by pre-deleting existing `Chunk` nodes and using `CREATE` instead of `MERGE+SET` when `use_vector_index=True`
- **Null-guard in `upsert_relations` and `delete`** — `lookup_relation` result is now checked for `None` before unpacking; falls back to `("Entity", "LINKS", "Entity")` if the label is not found
- **`strict_schema=True` ignored when `has_structured_schema=False`** — the flag is now clamped to `False` and a warning is logged

## [2026-03-27]

### Added
- `docs/api/README.md` — hand-written API reference for `LadybugGraphStore` and `LadybugPropertyGraphStore` covering all public constructors, methods, and properties
- `docs/notebooks/property_graph_ladybug.ipynb` — property graph example notebook using `PropertyGraphIndex` with structured schema, vector index, and combined graph+vector retrieval
- `docs/notebooks/LadybugGraphDemo.ipynb` — legacy knowledge graph example notebook using `KnowledgeGraphIndex` with pyvis visualization
- `docs/notebooks/ladybuggraph_draw.html` — pre-rendered interactive pyvis graph visualization
- `docs/notebooks/README.md` — overview of both notebooks and when to use each

### Changed
- PyPI package name set to `llama-index-graph-stores-ladybug` for consistency with other llama-index graph store integrations; GitHub repo remains `llama-index-ladybug`
- Bumped initial package version to `0.15.2` to align with the minimum supported `real-ladybug` version
- Added `nbstripout>=0.9.0` to dev dependencies and `.pre-commit-config.yaml` with an `nbstripout` hook so notebook outputs are always stripped before commits, regardless of whether Cursor has rerun them

### Fixed
- Upgraded `real-ladybug` dependency from `0.12.0` to `>=0.15.2,<0.16` to resolve a native access violation crash when calling `CREATE_VECTOR_INDEX` on Windows with Python 3.13
- Fixed deprecation warning in `base.py` by replacing separate `prepare()` + `execute()` calls with a single `execute()` call as required by the updated `real-ladybug` API
