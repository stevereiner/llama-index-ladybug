# API Reference

## `LadybugGraphStore`

```bash
uv pip install llama-index-ladybug
```

```python
from llama_index.graph_stores.ladybug import LadybugGraphStore
```

Legacy graph store using the `KnowledgeGraphIndex` API. Stores triplets (subject, predicate, object) in a Ladybug database. Use `LadybugPropertyGraphStore` for new projects.

### Constructor

```python
LadybugGraphStore(
    database: lb.Database,
    node_table_name: str = "entity",
    rel_table_name: str = "links",
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `database` | `lb.Database` | — | An open Ladybug `Database` instance |
| `node_table_name` | `str` | `"entity"` | Name of the node table in the database |
| `rel_table_name` | `str` | `"links"` | Name of the relationship table in the database |

### Class Methods

#### `from_persist_dir`

```python
@classmethod
LadybugGraphStore.from_persist_dir(
    persist_dir: str,
    node_table_name: str = "entity",
    rel_table_name: str = "links",
) -> LadybugGraphStore
```

Load a graph store from an existing Ladybug database directory.

#### `from_dict`

```python
@classmethod
LadybugGraphStore.from_dict(config_dict: dict) -> LadybugGraphStore
```

Initialize from a configuration dictionary. Keys correspond to constructor parameters.

### Instance Methods

#### `init_schema`

```python
init_schema() -> None
```

Creates the node and relationship tables if they do not already exist. Called automatically on construction.

#### `get`

```python
get(subj: str) -> List[List[str]]
```

Returns all `[predicate, object]` pairs for a given subject node ID.

#### `get_rel_map`

```python
get_rel_map(
    subjs: Optional[List[str]] = None,
    depth: int = 2,
    limit: int = 30,
) -> Dict[str, List[List[str]]]
```

Returns a depth-aware relationship map. Each key is a subject ID; values are lists of paths `[subj, pred, ..., obj]`.

#### `upsert_triplet`

```python
upsert_triplet(subj: str, rel: str, obj: str) -> None
```

Inserts a `(subject, predicate, object)` triplet, creating nodes if they do not exist. No-ops if the triplet already exists.

#### `delete`

```python
delete(subj: str, rel: str, obj: str) -> None
```

Deletes a specific triplet and removes orphaned nodes (nodes with no remaining edges).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `client` | `lb.Connection` | The underlying Ladybug connection object |

---

## `LadybugPropertyGraphStore`

```bash
uv pip install llama-index-ladybug
```

```python
from llama_index.graph_stores.ladybug import LadybugPropertyGraphStore
```

Property graph store using the `PropertyGraphIndex` API. Supports structured schemas, vector similarity search, and Cypher queries.

### Constructor

```python
LadybugPropertyGraphStore(
    db: lb.Database,
    relationship_schema: Optional[List[Tuple[str, str, str]]] = None,
    has_structured_schema: bool = False,
    sanitize_query_output: bool = True,
    use_vector_index: bool = True,
    embed_model: Optional[Any] = None,
    embed_dimension: Optional[int] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `lb.Database` | — | An open Ladybug `Database` instance |
| `relationship_schema` | `List[Tuple[str, str, str]]` | `None` | Required when `has_structured_schema=True`. List of `(head, relation, tail)` triples that define valid relationship types |
| `has_structured_schema` | `bool` | `False` | If `True`, enforces the provided `relationship_schema` on all upserted nodes and relations |
| `sanitize_query_output` | `bool` | `True` | If `True`, sanitizes query results to remove internal Ladybug metadata |
| `use_vector_index` | `bool` | `True` | If `True`, creates and uses a Ladybug HNSW vector index on `Chunk` nodes for similarity search |
| `embed_model` | embedding model | `None` | Embedding model instance used to auto-detect `embed_dimension` and for vector queries |
| `embed_dimension` | `int` | `None` | Explicit embedding dimension. Used as fallback if auto-detection from `embed_model` fails |

**Notes:**
- If `use_vector_index=True` and `embed_model` is provided, the embedding dimension is auto-detected by running a test embedding.
- If `has_structured_schema=True` and `relationship_schema=None`, a `ValueError` is raised.
- The `Chunk` node type is always added to the schema regardless of `has_structured_schema`.

### Instance Methods

#### `init_schema`

```python
init_schema() -> None
```

Creates all node tables, entity tables, and relationship tables in the database based on the schema. Called automatically on construction.

#### `upsert_nodes`

```python
upsert_nodes(nodes: List[LabelledNode]) -> None
```

Inserts or updates a list of `EntityNode` or `ChunkNode` objects. After upserting chunk nodes with embeddings, automatically creates the vector index if `use_vector_index=True`.

#### `upsert_relations`

```python
upsert_relations(relations: List[Relation]) -> None
```

Inserts or updates a list of `Relation` objects. Also creates `MENTIONS` edges from `Chunk` nodes to the related entities.

#### `get`

```python
get(
    properties: Optional[dict] = None,
    ids: Optional[List[str]] = None,
) -> List[LabelledNode]
```

Retrieves nodes matching the given `ids`. Returns a mix of `ChunkNode` and `EntityNode` objects.

#### `get_triplets`

```python
get_triplets(
    entity_names: Optional[List[str]] = None,
    relation_names: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
) -> List[Triplet]
```

Returns `[EntityNode, Relation, EntityNode]` triplets filtered by entity names, relation names, or node IDs.

#### `get_rel_map`

```python
get_rel_map(
    graph_nodes: List[LabelledNode],
    depth: int = 2,
    limit: int = 30,
    ignore_rels: Optional[List[str]] = None,
) -> List[List[Any]]
```

Returns relationship paths starting from the given nodes up to `depth` hops. Filters out relation types listed in `ignore_rels`.

#### `structured_query`

```python
structured_query(
    query: str,
    param_map: Optional[Dict[str, Any]] = None,
) -> Any
```

Executes a raw Cypher query against the database. Returns a list of dicts keyed by column name.

#### `vector_query`

```python
vector_query(
    query: VectorStoreQuery,
    **kwargs,
) -> Tuple[List[LabelledNode], List[float]]
```

Performs HNSW vector similarity search on `Chunk` nodes using the `chunk_embedding_index`. Returns `(nodes, similarities)` sorted by descending similarity. Requires `use_vector_index=True`.

#### `delete`

```python
delete(
    entity_names: Optional[List[str]] = None,
    relation_names: Optional[List[str]] = None,
    properties: Optional[dict] = None,
    ids: Optional[List[str]] = None,
) -> None
```

Deletes nodes and/or relationships matching any of the provided filters.

#### `get_schema`

```python
get_schema() -> dict
```

Returns the current schema of the graph as a dictionary with keys `node_props`, `rel_props`, and `relationships`.

Example output:
```python
{
    "node_props": {
        "Chunk": [{"property": "id", "type": "STRING"}, ...],
        "Entity": [{"property": "name", "type": "STRING"}, ...]
    },
    "rel_props": {
        "WORKS_AT": [{"property": "label", "type": "STRING"}]
    },
    "relationships": [
        {"start": "PERSON", "type": "WORKS_AT", "end": "ORGANIZATION"}
    ]
}
```

#### `refresh_vector_index`

```python
refresh_vector_index() -> None
```

Drops and recreates the `chunk_embedding_index` vector index. Useful after bulk data changes.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `client` | `lb.Connection` | The underlying Ladybug connection object |
