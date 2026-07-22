from typing import Any, Dict, List, Optional, Tuple
import logging
import llama_index.graph_stores.ladybug.utils as utils
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.graph_stores.utils import value_sanitize
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

import ladybug as lb

# Threshold for max number of returned triplets
LIMIT = 100
Triple = Tuple[str, str, str]
logger = logging.getLogger(__name__)


def _ensure_windows_vector_ext_deps() -> None:
    """Make the downloaded VECTOR extension's DLL dependencies loadable on Windows.

    The `vector` extension (downloaded by ``INSTALL vector``) imports
    ``libssl-3-x64.dll`` / ``libcrypto-3-x64.dll`` by their plain names. The ladybug
    wheel ships those OpenSSL DLLs in ``ladybug.libs`` but *delvewheel-mangled*
    (``libssl-3-x64-<hash>.dll``), so the engine's native ``LoadLibrary`` can't resolve
    the plain names and ``LOAD vector`` fails with WinError 126 ("The specified module
    could not be found") — unless an unmangled copy happens to be on ``PATH`` (e.g. from
    Git's ``mingw64\\bin``). Stage unmangled copies in a cache dir and put it on the DLL
    search path so ``LOAD vector`` works regardless of how Python was launched.

    Best-effort, Windows-only, no-op elsewhere. (Really an upstream ladybug packaging
    gap; this is a defensive shim.)
    """
    import sys

    if not sys.platform.startswith("win"):
        return
    try:
        import os
        import pathlib
        import shutil

        libs = pathlib.Path(lb.__file__).resolve().parent.parent / "ladybug.libs"
        if not libs.is_dir():
            return
        cache = pathlib.Path(os.path.expanduser("~")) / ".lbdb" / "dll_shim"
        cache.mkdir(parents=True, exist_ok=True)
        staged = False
        for plain in ("libssl-3-x64.dll", "libcrypto-3-x64.dll"):
            dst = cache / plain
            if dst.exists():
                staged = True
                continue
            hits = list(libs.glob(plain[:-4] + "-*.dll"))
            if hits:
                shutil.copy(hits[0], dst)
                staged = True
        if staged:
            os.environ["PATH"] = str(cache) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(cache))
            except (OSError, AttributeError):
                pass
    except Exception as e:  # never block store init on the shim
        logger.debug("Ladybug: could not stage Windows vector-ext DLLs: %s", e)


class LadybugPropertyGraphStore(PropertyGraphStore):
    """
    Ladybug Property Graph Store.

    Supports three schema modes controlled by ``has_structured_schema`` and
    ``strict_schema``:

    **Unstructured** (``has_structured_schema=False``, default)
        No ontology is enforced. All LLM-extracted entity and relation types are
        stored freely. The database contains two node tables (``Entity``, ``Chunk``)
        and two relation tables (``LINKS``, ``MENTIONS``).

    **Structured + non-strict** (``has_structured_schema=True``, ``strict_schema=False``)
        A ``relationship_schema`` defines the expected node/relation types. Those
        tables are created as-is. In addition, an ``Entity`` node table and a
        ``LINKS`` catch-all relation table are also created so that LLM-extracted
        types that fall *outside* the ontology are stored rather than dropped.
        The schema can therefore expand beyond what was declared at construction time.

    **Structured + strict** (``has_structured_schema=True``, ``strict_schema=True``)
        Only the node and relation types declared in ``relationship_schema`` are
        created. Any entity or relation that cannot be matched to the schema is
        silently dropped at ingest time — no ``Entity``/``LINKS`` overflow tables
        are created.

    In all modes a ``Chunk`` node table and a ``MENTIONS`` relation table are
    always present.

    Ladybug can be installed with::

        uv pip install ladybug
    """

    def __init__(
        self,
        db: lb.Database,
        relationship_schema: Optional[List[Tuple[str, str, str]]] = None,
        has_structured_schema: Optional[bool] = False,
        strict_schema: bool = False,
        sanitize_query_output: Optional[bool] = True,
        use_vector_index: bool = True,
        embed_model: Optional[Any] = None,
        embed_dimension: Optional[int] = None,
    ) -> None:
        self.db = db
        self.connection = lb.Connection(self.db)
        self.async_connection = lb.AsyncConnection(self.db)
        self.use_vector_index = use_vector_index
        self.strict_schema = strict_schema

        # Initialize embedding dimension with auto-detection and fallback logic
        self.embed_dimension = self._initialize_embedding_dimension(
            embed_model, embed_dimension
        )

        # Install and load vector extension if using vector indexes.
        # As of Ladybug 0.18.x vector indexing lives in a loadable VECTOR extension
        # (CREATE_VECTOR_INDEX is no longer a core function). INSTALL downloads the
        # extension once (needs network the first time, persisted to disk after) and can
        # error if it's already installed or unreachable; LOAD activates it per-connection
        # and must run every session. Keep them separate so a benign INSTALL error doesn't
        # skip LOAD, and log (don't silently swallow) so a genuine load failure is visible
        # instead of surfacing later as a confusing "CREATE_VECTOR_INDEX is not defined".
        if self.use_vector_index:
            _ensure_windows_vector_ext_deps()
            try:
                self.connection.execute("INSTALL vector;")
            except RuntimeError as e:
                logger.debug(
                    "Ladybug vector extension INSTALL skipped (already installed or "
                    "unreachable): %s",
                    e,
                )
            try:
                self.connection.execute("LOAD vector;")
            except RuntimeError as e:
                logger.warning(
                    "Ladybug vector extension failed to LOAD — vector indexing will be "
                    "unavailable. Install it manually in a Ladybug CLI session with "
                    "`INSTALL vector; LOAD vector;` (needs network the first time). Error: %s",
                    e,
                )

        if has_structured_schema:
            if relationship_schema is None:
                logger.warning(
                    "LadybugPropertyGraphStore: has_structured_schema=True but no relationship_schema — falling back to unstructured mode."
                )
                has_structured_schema = False
                self.strict_schema = False
                relationship_schema = [("Entity", "LINKS", "Entity")]
            else:
                self.validate_relationship_schema(relationship_schema)
        else:
            # Use a generic schema with node types of 'Entity' if no schema is required
            if strict_schema:
                logger.warning(
                    "LadybugPropertyGraphStore: strict_schema=True ignored when has_structured_schema=False."
                )
                self.strict_schema = False
            relationship_schema = [("Entity", "LINKS", "Entity")]

        # Sanitize schema labels — rename Ladybug reserved type names (e.g. TIME,
        # DATE) that cause native crashes when used as node table names.
        # Node/entity labels use safe_label (_TYPE suffix).
        # Relation labels use safe_rel_label (_REL suffix) — must differ from node
        # table names since Ladybug does not allow a rel table and node table to share a name.
        relationship_schema = [
            (utils.safe_label(src), utils.safe_rel_label(rel), utils.safe_label(dst))
            for src, rel, dst in relationship_schema
        ]
        self.relationship_schema = relationship_schema
        self.entities = self.get_entities()
        self.has_structured_schema = has_structured_schema
        self.entities.extend(
            ["Chunk"]
        )  # Always include Chunk as an entity type, in all schemas
        self.sanitize_query_output = sanitize_query_output
        self.structured_schema = {}

        # supports_vector_queries = True: VectorContextRetriever calls avector_query()
        # (async, runs in Ladybug's AsyncConnection thread pool) instead of an external
        # in-memory SimpleVectorStore that is empty on restart.
        if use_vector_index:
            self.supports_vector_queries = True

        self.init_schema()

    def _checkpoint(self) -> None:
        """Flush WAL to disk, logging at debug level if it fails."""
        try:
            self.connection.execute("CHECKPOINT;")
        except Exception as _e:
            logger.debug("Ladybug CHECKPOINT failed: %s", _e)

    def init_schema(self) -> None:
        """Initialize schema if the required tables do not exist.

        Schema per mode:
        - structured + strict:     schema node types + Chunk; schema rels + MENTIONS only
        - structured + non-strict: schema node types + Chunk + Entity; schema rels + MENTIONS + LINKS
        - unstructured:            Entity + Chunk only; LINKS + MENTIONS only

        NOTE: if you see a 'Table TIME_TYPE does not exist' or similar Binder exception
        on startup, the database was created with an older schema version. Delete the
        ./ladybug directory and re-ingest all documents.
        """
        utils.create_chunk_node_table(
            self.connection, embedding_dimension=self.embed_dimension
        )

        if self.has_structured_schema:
            # Create the ontology-defined node tables
            utils.create_entity_node_tables(self.connection, entities=self.entities)
            # Filter out any user-supplied LINKS entry — managed below per mode
            schema_without_links = [t for t in self.relationship_schema if t[1] != "LINKS"]
            utils.create_relation_tables(
                self.connection,
                self.entities,
                relationship_schema=schema_without_links,
            )
            if not self.strict_schema:
                # Non-strict: Entity table catches off-schema overflow; LINKS connects them
                utils.create_entity_node_tables(self.connection, entities=["Entity"])
                all_types = list(set(self.entities + ["Entity"]))
                utils.create_catch_all_links_table(self.connection, all_types)
            # strict: no Entity table, no LINKS — off-schema entities/relations are dropped
        else:
            # Unstructured: only Entity + Chunk nodes; only LINKS + MENTIONS relations
            utils.create_entity_node_tables(self.connection, entities=["Entity"])
            utils.create_catch_all_links_table(self.connection, ["Entity"])

    def validate_relationship_schema(self, relationship_schema: List[Triple]) -> None:
        # Check that validation schema is a list of tuples as required by Kùzu for relationships
        if not all(isinstance(item, tuple) for item in relationship_schema):
            raise ValueError(
                "Please specify the relationship schema as "
                "a list of tuples, for example: [('PERSON', 'IS_CEO_OF', 'ORGANIZATION')]"
            )

    @property
    def client(self) -> lb.Connection:
        return self.connection

    def get_entities(self) -> List[str]:
        return sorted(
            set(
                [rel[0] for rel in self.relationship_schema]
                + [rel[2] for rel in self.relationship_schema]
            )
        )

    def _initialize_embedding_dimension(
        self, embed_model: Optional[Any], embedding_dimension: Optional[int]
    ) -> Optional[int]:
        """
        Initialize embedding dimension using auto-detection and fallback logic.

        Args:
            embed_model: Optional embedding model for auto-detection
            embedding_dimension: Optional manual dimension specification

        Returns:
            Detected or specified embedding dimension, or None if unavailable

        """
        if embed_model is not None:
            # Try auto-detection first
            detected_dim = self._detect_embedding_dimension(embed_model)
            if detected_dim is not None:
                print(f"Auto-detected embedding dimension: {detected_dim}")
                return detected_dim
            elif embedding_dimension is not None:
                # Fall back to manual specification if auto-detection fails
                print(
                    f"Using manually specified embedding dimension: {embedding_dimension}"
                )
                return embedding_dimension
            else:
                # Neither auto-detection nor manual specification available
                print(
                    "Warning: Could not determine embedding dimension. Vector indexing may not work properly."
                )
                return None
        else:
            # No embed_model provided, use manual specification
            if embedding_dimension is not None:
                print(
                    f"Using manually specified embedding dimension: {embedding_dimension}"
                )
            return embedding_dimension

    def _detect_embedding_dimension(self, embed_model: Any) -> Optional[int]:
        """
        Detect embedding dimension by creating a test embedding.

        Args:
            embed_model: The embedding model instance

        Returns:
            Detected dimension or None if cannot be determined

        """
        try:
            test_embedding = embed_model.get_text_embedding("hello")
            if isinstance(test_embedding, list) and len(test_embedding) > 0:
                return len(test_embedding)
        except Exception:
            print(
                "Error: Could not detect embedding dimension from model. Please specify it manually via the `embedding_dimension` parameter."
            )  # noqa: E501

        return None

    def _create_vector_index(self, table_name: str) -> None:
        """Create a vector index for the embedding column of a table."""
        if not self.use_vector_index or table_name != "Chunk":
            return

        # Cannot create a vector index on DOUBLE[] (variable-length) — requires
        # a fixed-size DOUBLE[N] column. Skip silently if no dimension was set.
        if not self.embed_dimension:
            logger.debug(
                "Ladybug: skipping vector index creation — embed_dimension not set"
            )
            return

        # Check if chunk_embedding_index already exists
        existing_indexes_result = self.connection.execute(
            "CALL SHOW_INDEXES() RETURN *"
        )
        for row in existing_indexes_result:
            if len(row) > 1 and row[1] == "chunk_embedding_index":
                return

        # Check if table has any data - Ladybug requires data before creating vector index
        count_result = self.connection.execute(
            f"MATCH (n:{utils.quote_id(table_name)}) RETURN COUNT(n)"
        )
        if not any(int(row[0]) > 0 for row in count_result):
            return

        # Create vector index for Chunk table
        self.connection.execute("""
        CALL CREATE_VECTOR_INDEX(
            'Chunk',
            'chunk_embedding_index',
            'embedding',
            metric := 'cosine'
        )
        """)

    def _ensure_vector_indexes(self) -> None:
        """Ensure vector indexes are created for Chunk table only."""
        if not self.use_vector_index:
            return
        # Only create index for Chunk table since these have larger blobs of text
        # This makes the workflow easier to manage as a whole
        self._create_vector_index("Chunk")

    def refresh_vector_index(self) -> None:
        """Drop and recreate the vector index for Chunk table."""
        index_name = "chunk_embedding_index"
        # Drop existing index if it exists
        try:
            self.connection.execute("CALL DROP_VECTOR_INDEX('Chunk', 'chunk_embedding_index')")
            print(f"Dropped vector index: {index_name}")
        except Exception:
            # Index may not exist, which is fine
            pass

        # Recreate the index
        self._create_vector_index("Chunk")
        print(f"Created vector index: {index_name}")

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        entity_list: List[EntityNode] = []
        chunk_list: List[ChunkNode] = []
        node_tables = self.connection._get_node_table_names()
        rel_tables = self.connection._get_rel_table_names()

        for item in nodes:
            if isinstance(item, EntityNode):
                entity_list.append(item)
            elif isinstance(item, ChunkNode):
                chunk_list.append(item)
        for chunk in chunk_list:
            if self.use_vector_index:
                # Ladybug HNSW index supports incremental inserts — new rows are picked
                # up automatically without DROP/CREATE.  The only constraint is that
                # Ladybug raises "Cannot set property … Try delete and then insert" when
                # you UPDATE an embedding column that is covered by a vector index.
                # Safe strategy (never touches the global index):
                #   1. DETACH DELETE the chunk ONLY if it belongs to the same document
                #      (same ref_doc_id).  This guards against LlamaIndex assigning the
                #      same chunk UUID to two different documents (e.g. cmispress.txt and
                #      cmispress.pdf share the same filename stem → same doc_id → same
                #      chunk UUID).  Deleting a chunk from a *different* doc would wipe
                #      its MENTIONS permanently (they'd never be re-created this pass).
                #   2. CREATE the chunk fresh with embedding inline.
                # For a brand-new chunk (step 1 is a no-op), CREATE inserts into the
                # live HNSW index directly — no global drop/rebuild needed.
                _ref_doc_id = chunk.properties.get("ref_doc_id")
                _skip_chunk = False
                try:
                    if _ref_doc_id:
                        # Only delete this chunk if it belongs to the same document.
                        # Matching on ref_doc_id prevents wiping a same-UUID chunk from
                        # a different document (e.g. cmispress.txt vs cmispress.pdf share
                        # the same LlamaIndex doc_id -> same chunk UUID).
                        self.connection.execute(
                            "MATCH (c:Chunk {id: $id, ref_doc_id: $ref_doc_id}) DETACH DELETE c",
                            parameters={"id": chunk.id_, "ref_doc_id": _ref_doc_id},
                        )
                        # Check if a chunk with this id still exists (UUID collision:
                        # belongs to a different doc — do not overwrite it).
                        _exists_res = self.connection.execute(
                            "MATCH (c:Chunk {id: $id}) RETURN count(c) AS n",
                            parameters={"id": chunk.id_},
                        )
                        _exists_rows = _exists_res.get_as_df() if hasattr(_exists_res, "get_as_df") else None
                        if _exists_rows is not None and not _exists_rows.empty and int(_exists_rows.iloc[0]["n"]) > 0:
                            logger.warning(
                                "Ladybug: chunk %s UUID collision (belongs to different doc) — skipping CREATE",
                                chunk.id_[:8],
                            )
                            _skip_chunk = True
                        else:
                            logger.debug("Ladybug: pre-delete Chunk %s (DETACH DELETE, ref_doc=%s)", chunk.id_[:8], _ref_doc_id[:8])
                    else:
                        self.connection.execute(
                            "MATCH (c:Chunk {id: $id}) DETACH DELETE c",
                            parameters={"id": chunk.id_},
                        )
                        logger.debug("Ladybug: pre-delete Chunk %s (DETACH DELETE, no ref_doc)", chunk.id_[:8])
                except Exception as _del_err:
                    logger.debug("Ladybug: pre-delete Chunk %s skipped: %s", chunk.id_[:8], _del_err)
                if _skip_chunk:
                    continue
                upsert_chunk_node_query = """
                    CREATE (c:Chunk {
                        id: $id,
                        text: $text,
                        label: $label,
                        embedding: $embedding,
                        ref_doc_id: $ref_doc_id,
                        creation_date: date($creation_date),
                        last_modified_date: date($last_modified_date),
                        file_name: $file_name,
                        file_path: $file_path,
                        file_size: $file_size,
                        file_type: $file_type
                    })
                    """
            else:
                # No vector index — safe to MERGE+SET directly.
                upsert_chunk_node_query = """
                    MERGE (c:Chunk {id: $id})
                      SET c.text = $text,
                          c.label = $label,
                          c.embedding = $embedding,
                          c.ref_doc_id = $ref_doc_id,
                          c.creation_date = date($creation_date),
                          c.last_modified_date = date($last_modified_date),
                          c.file_name = $file_name,
                          c.file_path = $file_path,
                          c.file_size = $file_size,
                          c.file_type = $file_type
                    """

            self.connection.execute(
                upsert_chunk_node_query,
                parameters={
                    "id": chunk.id_,
                    "text": chunk.text.strip(),
                    "label": chunk.label,
                    "embedding": chunk.embedding,
                    "ref_doc_id": chunk.properties.get("ref_doc_id"),
                    "creation_date": chunk.properties.get("creation_date"),
                    "last_modified_date": chunk.properties.get("last_modified_date"),
                    "file_name": chunk.properties.get("file_name"),
                    "file_path": chunk.properties.get("file_path"),
                    "file_size": chunk.properties.get("file_size"),
                    "file_type": chunk.properties.get("file_type"),
                },
            )

        if chunk_list and self.use_vector_index:
            # Ensure the HNSW index exists (creates it on first ingest; no-op if
            # already present since _create_vector_index checks SHOW_INDEXES first).
            self._create_vector_index("Chunk")

        # Track labels already warned/created this call to avoid repeat DDL and log spam
        _processed_labels: set = set()
        # In strict mode, track which node ids were skipped so upsert_relations
        # can skip any relation whose source or target was not stored.
        _skipped_ids: set = set()

        for _entity_idx, entity in enumerate(entity_list):
            if entity.label in node_tables:
                entity_label = entity.label
            elif self.has_structured_schema and not self.strict_schema:
                # Non-strict structured mode: create typed table on the fly for
                # new entity types the LLM extracted that aren't in the schema.
                entity_label = utils.safe_label(entity.label, existing_rel_tables=rel_tables)
                if entity_label not in _processed_labels:
                    logger.debug(
                        f"Ladybug: creating new node table '{entity_label}' "
                        f"(not in schema, strict_schema=False)"
                    )
                    utils.create_entity_node_tables(self.connection, entities=[entity_label])
                    # Also extend MENTIONS to cover this new type.
                    # Use ADD IF NOT EXISTS — plain ADD FROM drops and recreates the
                    # internal edge storage, wiping all existing MENTIONS edges.
                    try:
                        self.connection.execute(
                            f"ALTER TABLE MENTIONS ADD IF NOT EXISTS FROM Chunk TO {utils.quote_id(entity_label)}"
                        )
                        logger.debug("Ladybug: extended MENTIONS for new type '%s'", entity_label)
                    except Exception as _alt_err:
                        logger.debug("Ladybug: MENTIONS ALTER for '%s' failed: %s", entity_label, _alt_err)
                    node_tables.append(entity_label)
                    _processed_labels.add(entity_label)
            elif self.has_structured_schema and self.strict_schema:
                # Strict mode: skip entities whose type is not in the schema entirely.
                if entity.label not in _processed_labels:
                    logger.warning(
                        f"Ladybug: skipping entity '{entity.name}' — label '{entity.label}' "
                        f"not in schema (strict_schema=True)"
                    )
                    _processed_labels.add(entity.label)
                _skipped_ids.add(entity.name)
                continue
            else:
                # Unstructured mode: fall back to Entity table
                entity_label = "Entity"

            upsert_entity_node_query = f"""
                MERGE (e:{utils.quote_id(entity_label)} {{id: $id}})
                SET e.label = $label,
                    e.name = $name,
                    e.creation_date = date($creation_date),
                    e.last_modified_date = date($last_modified_date),
                    e.file_name = $file_name,
                    e.file_path = $file_path,
                    e.file_size = $file_size,
                    e.file_type = $file_type,
                    e.triplet_source_id = $triplet_source_id,
                    e.ref_doc_id = $ref_doc_id
                """

            self.connection.execute(
                upsert_entity_node_query,
                parameters={
                    "id": entity.name,
                    "label": entity.label,
                    "name": entity.name,
                    "creation_date": entity.properties.get("creation_date"),
                    "last_modified_date": entity.properties.get("last_modified_date"),
                    "file_name": entity.properties.get("file_name"),
                    "file_path": entity.properties.get("file_path"),
                    "file_size": entity.properties.get("file_size"),
                    "file_type": entity.properties.get("file_type"),
                    "triplet_source_id": entity.properties.get("triplet_source_id"),
                    "ref_doc_id": entity.properties.get("ref_doc_id") or entity.properties.get("doc_id"),
                },
            )
            # Cache id -> actual table name so upsert_relations needs no DB lookups.
            # Cache by both name and id since Relation.source_id may be either.
            if not hasattr(self, "_node_label_cache"):
                self._node_label_cache: dict = {}
            self._node_label_cache[entity.name] = entity_label
            if entity.id != entity.name:
                self._node_label_cache[entity.id] = entity_label

        # Persist the skipped id set so upsert_relations can filter relations
        # whose endpoints were never stored (strict mode only).
        if _skipped_ids:
            if not hasattr(self, "_skipped_node_ids"):
                self._skipped_node_ids: set = set()
            self._skipped_node_ids.update(_skipped_ids)
        # Final checkpoint after all entity upserts
        self._checkpoint()
        stored_entities = len(entity_list) - len(_skipped_ids)
        logger.info(
            f"Ladybug upsert_nodes done: {len(chunk_list)} chunks, "
            f"{stored_entities} entities ({len(_skipped_ids)} skipped)"
        )

    def _get_node_label(self, node_id: str) -> str:
        """Look up the actual stored node table label for a given node id.
        Checks the in-memory cache populated during upsert_nodes first —
        avoids N table scans per node when called from upsert_relations.
        Falls back to DB scan if not in cache (e.g. nodes from a prior session).
        """
        if hasattr(self, "_node_label_cache") and node_id in self._node_label_cache:
            return self._node_label_cache[node_id]
        # Cache miss — log at debug so we can diagnose label-lookup failures
        logger.debug("Ladybug _get_node_label: cache miss for '%s', scanning tables", node_id[:60] if len(node_id) > 60 else node_id)
        node_tables = self.connection._get_node_table_names()
        for tbl in node_tables:
            if tbl == "Chunk":
                continue
            try:
                res = self.connection.execute(
                    f"MATCH (n:{utils.quote_id(tbl)} {{id: $id}}) RETURN n.id LIMIT 1",
                    parameters={"id": node_id},
                ).get_as_df()
                if len(res) > 0:
                    if not hasattr(self, "_node_label_cache"):
                        self._node_label_cache = {}
                    self._node_label_cache[node_id] = tbl
                    return tbl
            except Exception:
                pass
        return "Entity"

    def upsert_relations(self, relations: List[Relation]) -> None:
        # Pre-build a node_id -> table_label cache to avoid repeated lookups
        # for the same node across multiple relations.
        _label_cache: dict = {}
        _skipped = getattr(self, "_skipped_node_ids", set())
        # Track (rel_table, src_type, dst_type) tuples already ensured this call
        # to avoid redundant DDL for the same type pair.
        _ensured_rel_pairs: set = set()
        _mentions_count: int = 0
        # Fetch current node tables so safe_rel_label can avoid name collisions
        # with dynamically-created node tables (e.g. FEATURE node table vs FEATURE rel).
        node_tables = self.connection._get_node_table_names()

        def _node_label(node_id: str) -> str:
            if node_id not in _label_cache:
                _label_cache[node_id] = self._get_node_label(node_id)
            return _label_cache[node_id]

        for _rel_idx, rel in enumerate(relations):
            # In strict mode, drop any relation whose endpoints were not stored
            if _skipped and (rel.source_id in _skipped or rel.target_id in _skipped):
                logger.debug(
                    f"Ladybug: skipping relation '{rel.label}' "
                    f"({rel.source_id} -> {rel.target_id}) — endpoint not in schema"
                )
                continue
            if self.has_structured_schema:
                result = utils.lookup_relation(
                    rel.label, self.relationship_schema
                )
                if result is None:
                    # LLM produced a relation label not in the schema (case mismatch or
                    # out-of-schema label). Try case-insensitive match first.
                    result = utils.lookup_relation(
                        rel.label.upper(), self.relationship_schema
                    )

                # Resolve actual stored node table labels (needed in structured mode
                # where nodes live in typed tables like EMPLOYEE, ORGANIZATION, etc.)
                src_label = _node_label(rel.source_id)
                dst_label = _node_label(rel.target_id)
                logger.debug(
                    "Ladybug upsert_relations: rel='%s' src='%s'(%s) dst='%s'(%s)",
                    rel.label, rel.source_id[:40], src_label, rel.target_id[:40], dst_label,
                )

                if result is None:
                    if not self.strict_schema:
                        # Non-strict: create/extend a named typed rel table on the fly.
                        # ensure_rel_table uses CREATE IF NOT EXISTS + ALTER IF NOT EXISTS
                        # so it's idempotent and handles multi-type pairs correctly.
                        rel_tbl_name = utils.safe_rel_label(
                            rel.label.upper().replace(" ", "_"),
                            existing_node_tables=node_tables,
                        )
                        pair_key = (rel_tbl_name, src_label, dst_label)
                        if pair_key not in _ensured_rel_pairs:
                            logger.debug(
                                f"Ladybug: creating rel table '{rel_tbl_name}' "
                                f"({src_label}->{dst_label}, not in schema)"
                            )
                            utils.ensure_rel_table(
                                self.connection, rel_tbl_name, src_label, dst_label
                            )
                            # Checkpoint after each new DDL to flush WAL and avoid
                            # native Ladybug crash on large batches of new relation tables.
                            self._checkpoint()
                            _ensured_rel_pairs.add(pair_key)
                        src, dst = src_label, dst_label
                        use_typed_match = True
                    else:
                        # Strict: drop off-schema relations entirely
                        logger.debug(
                            f"Ladybug: skipping relation '{rel.label}' — not in schema "
                            f"(strict_schema=True)"
                        )
                        continue
                else:
                    src, rel_tbl_name, dst = result
                    # Schema rels: ensure the actual FROM/TO pair is registered in the
                    # rel table (schema may define multiple src/dst types per rel label;
                    # only the first pair is created by CREATE REL TABLE — subsequent
                    # actual combinations need ALTER TABLE ADD before MERGE).
                    pair_key = (rel_tbl_name, src_label, dst_label)
                    if pair_key not in _ensured_rel_pairs:
                        utils.ensure_rel_table(
                            self.connection, rel_tbl_name, src_label, dst_label
                        )
                        _ensured_rel_pairs.add(pair_key)
                    src, dst = src_label, dst_label
                    use_typed_match = True
            else:
                # Unstructured mode: all nodes stored in generic Entity table.
                # The node's type name lives in its label property, not the table name.
                src, rel_tbl_name, dst = "Entity", "LINKS", "Entity"
                src_label, dst_label = "Entity", "Entity"
                use_typed_match = True  # LINKS is multi-type; typed MATCH required

            # Connect entities to each other.
            # - Typed MATCH: LINKS (multi-type), dynamic off-schema rels (multi-FROM/TO)
            # - Label-free MATCH: schema-defined rels — nodes may be stored under a
            #   fallback table (e.g. Entity) so typed MATCH would fail
            try:
                if use_typed_match:
                    self.connection.execute(
                        f"""
                        MATCH (a:{utils.quote_id(src_label)} {{id: $source_id}}),
                              (b:{utils.quote_id(dst_label)} {{id: $target_id}})
                        MERGE (a)-[r:{utils.quote_id(rel_tbl_name)} {{label: $label}}]->(b)
                            SET r.triplet_source_id = $triplet_source_id
                        """,
                        parameters={
                            "source_id": rel.source_id,
                            "target_id": rel.target_id,
                            "triplet_source_id": rel.properties.get("triplet_source_id"),
                            "label": rel.label,
                        },
                    )
                else:
                    self.connection.execute(
                        f"""
                        MATCH (a) WHERE a.id = $source_id
                        MATCH (b) WHERE b.id = $target_id
                        MERGE (a)-[r:{utils.quote_id(rel_tbl_name)} {{label: $label}}]->(b)
                            SET r.triplet_source_id = $triplet_source_id
                        """,
                        parameters={
                            "source_id": rel.source_id,
                            "target_id": rel.target_id,
                            "triplet_source_id": rel.properties.get("triplet_source_id"),
                            "label": rel.label,
                        },
                    )
            except Exception as e:
                logger.warning(
                    f"Ladybug: skipping relation '{rel.label}' "
                    f"({rel.source_id} -> {rel.target_id}): {e}"
                )
                continue

            # Connect chunks to entities — MENTIONS is a multi-label rel table so
            # Ladybug requires typed node labels in this MERGE (label-free MATCH fails).
            tsid = rel.properties.get("triplet_source_id")
            if not tsid:
                continue  # No chunk linkage available for this relation
            try:
                self.connection.execute(
                    f"""
                    MATCH (a:{utils.quote_id(src_label)} {{id: $source_id}}),
                            (b:{utils.quote_id(dst_label)} {{id: $target_id}}),
                            (c:Chunk {{id: $triplet_source_id}})
                    MERGE (c)-[:MENTIONS]->(a)
                    MERGE (c)-[:MENTIONS]->(b)
                    """,
                    parameters={
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "triplet_source_id": tsid,
                    },
                )
                _mentions_count += 1
            except Exception as e:
                logger.warning(
                    f"Ladybug: skipping MENTIONS for relation '{rel.label}' "
                    f"({rel.source_id} -> {rel.target_id}): {e}"
                )

        # Final checkpoint
        self._checkpoint()
        # Count total MENTIONS in DB to verify integrity after upsert
        try:
            _total_res = self.connection.execute("MATCH (c:Chunk)-[:MENTIONS]->(e) RETURN count(*) AS n")
            _total_rows = list(_total_res)
            _total_mentions = _total_rows[0][0] if _total_rows else 0
            # Also count per chunk for diagnostics
            _per_chunk_res = self.connection.execute(
                "MATCH (c:Chunk)-[:MENTIONS]->(e) RETURN c.id AS cid, count(e) AS n ORDER BY n DESC LIMIT 10"
            )
            _per_chunk = [(row[0][:8], row[1]) for row in _per_chunk_res]
        except Exception:
            _total_mentions = -1
            _per_chunk = []
        logger.info(
            f"Ladybug upsert_relations done: {len(relations)} relations, "
            f"{_mentions_count} MENTIONS chunk-entity links (total MENTIONS in DB: {_total_mentions}, "
            f"per-chunk top10: {_per_chunk})"
        )

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        response = self.connection.execute(query, parameters=param_map)
        column_names = response.get_column_names()
        result = []
        for row in response:
            result.append(dict(zip(column_names, row)))

        if self.sanitize_query_output:
            result = value_sanitize(result)

        return result

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Perform vector similarity search on Chunk nodes.

        Returns entity nodes (not chunk nodes) so that VectorContextRetriever's
        scoring logic works correctly: it scores triplets by checking whether
        triplet[0].id or triplet[2].id appears in the returned node IDs.
        Each entity node carries the chunk's similarity score and has
        triplet_source_id set to the originating chunk ID so that
        add_source_text() can fetch and attach the chunk text.
        This mirrors how ArcadeDB and Neo4j integrate with VectorContextRetriever.

        Returns empty results gracefully when the vector index does not exist yet
        (before first ingest). Ladybug requires at least one row in a table before
        a vector index can be created; a search arriving before first ingest raises
        a ``Binder`` or ``chunk_embedding_index`` error.
        """
        try:
            return self._vector_query_impl(query, **kwargs)
        except Exception as _vec_err:
            err_str = str(_vec_err)
            if "chunk_embedding_index" in err_str or "Binder" in err_str:
                logger.debug(
                    "Ladybug vector_query: index not ready yet (%s), returning empty",
                    _vec_err,
                )
                return [], []
            raise

    def _vector_query_impl(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Internal implementation of vector_query — called by the public wrapper."""
        self._ensure_vector_indexes()

        # Step 1: vector search on Chunk nodes
        result = self.connection.execute(
            """
            CALL QUERY_VECTOR_INDEX(
                'Chunk',
                'chunk_embedding_index',
                $query_embedding,
                $top_k
            )
            RETURN node.id as id, distance
            ORDER BY distance
            """,
            parameters={
                "query_embedding": query.query_embedding,
                "top_k": query.similarity_top_k,
            },
        )

        # Collect (chunk_id, similarity) pairs
        chunk_scores: List[Tuple[str, float]] = []
        for row in result:
            node_id, distance = row[0], row[1]
            similarity = max(0.0, 1.0 - distance)
            chunk_scores.append((node_id, similarity))
            logger.debug(
                f"Ladybug vector_query: chunk {node_id[:8]}... "
                f"distance={distance:.4f} similarity={similarity:.4f}"
            )

        if not chunk_scores:
            logger.debug("Ladybug vector_query: returned 0 chunks")
            self._checkpoint()
            return [], []

        # Step 2: expand each chunk to its mentioned entities via MENTIONS,
        # keeping the chunk's similarity score for each entity.
        # This matches ArcadeDB/Neo4j behaviour where vector_query returns
        # entity nodes so VectorContextRetriever can score triplets correctly.
        chunk_ids = [c[0] for c in chunk_scores]
        chunk_score_map = dict(chunk_scores)

        # Query MENTIONS directly via connection.execute to avoid value_sanitize()
        # in structured_query() which silently drops lists of >= 128 rows.
        _entity_result = self.connection.execute(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            WHERE c.id IN $chunk_ids
            RETURN e.id AS entity_id, c.id AS chunk_id, e.name AS name, e, label(e) AS entity_label
            LIMIT 512
            """,
            parameters={"chunk_ids": chunk_ids},
        )
        _col_names = _entity_result.get_column_names()
        entity_rows = [dict(zip(_col_names, row)) for row in _entity_result]

        # Build entity nodes keyed by name (= LlamaIndex EntityNode.id), keeping
        # max score per entity. triplet_source_id set for add_source_text() lookup.
        # e._LABEL cannot be selected as a column (it's a Ladybug internal metadata field);
        # instead we return the whole node as 'e' and extract _LABEL from the dict.
        entity_score: dict = {}  # name -> (score, chunk_id, entity_id, label)
        for row in entity_rows:
            eid = row.get("entity_id")
            cid = row.get("chunk_id")
            name = row.get("name") or eid
            if not name or not cid:
                continue
            score = chunk_score_map.get(cid, 0.0)
            node_dict = row.get("e") or {}
            label = row.get("entity_label") or node_dict.get("_LABEL") or "Entity"
            if name not in entity_score or score > entity_score[name][0]:
                entity_score[name] = (score, cid, eid, label)

        if not entity_score:
            # No MENTIONS found — fall back to returning chunk nodes directly
            # (handles databases ingested before MENTIONS were stored)
            logger.debug(
                "Ladybug vector_query: no MENTIONS found, falling back to chunk nodes"
            )
            node_data = []
            for chunk_id, similarity in chunk_scores:
                chunk_result = self.structured_query(
                    "MATCH (n:Chunk {id: $node_id}) RETURN n.*",
                    param_map={"node_id": chunk_id},
                )
                if chunk_result:
                    record = chunk_result[0]
                    properties = {
                        k: v for k, v in record.items() if k not in ["n.id", "n.text"]
                    }
                    node = ChunkNode(
                        id_=record["n.id"],
                        text=record.get("n.text") or "",
                        properties=utils.remove_empty_values(properties),
                    )
                    node_data.append((node, similarity))
            node_data.sort(key=lambda x: x[1], reverse=True)
            nodes = [x[0] for x in node_data]
            similarities = [x[1] for x in node_data]
            logger.debug(
                f"Ladybug vector_query: returned {len(nodes)} chunk nodes (fallback)"
            )
            self._checkpoint()
            return nodes, similarities

        # Build EntityNode list sorted by score descending.
        # EntityNode.id == EntityNode.name (LlamaIndex design) so using name as key
        # ensures VectorContextRetriever's kg_ids.index(triplet[0].id) lookup succeeds
        # and triplet scores propagate correctly (same pattern as ArcadeDB / Neo4j).
        node_data = []
        for name, (score, chunk_id, entity_id, label) in entity_score.items():
            node = EntityNode(
                name=name,
                label=label,
                properties={"triplet_source_id": chunk_id},
            )
            node_data.append((node, score))

        node_data.sort(key=lambda x: x[1], reverse=True)
        nodes = [x[0] for x in node_data]
        similarities = [x[1] for x in node_data]

        logger.info(
            f"Ladybug vector_query: returned {len(nodes)} entity nodes "
            f"(from {len(chunk_scores)} chunks), "
            f"top score={similarities[0]:.4f}"
        )

        self._checkpoint()
        return nodes, similarities

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes from the property graph store."""
        cypher_statement = "MATCH (e) "

        parameters = {}
        if ids:
            cypher_statement += "WHERE e.id in $ids "
            parameters["ids"] = ids

        return_statement = "RETURN e.*"
        cypher_statement += return_statement
        result = self.structured_query(cypher_statement, param_map=parameters)
        result = result if result else []

        nodes = []
        for record in result:
            # Identify chunk nodes by the presence of text content or Chunk label.
            # Our Chunk table uses _LABEL="Chunk"; legacy check for "text_chunk" kept too.
            is_chunk = (
                record.get("e.label") == "text_chunk"
                or record.get("e._LABEL") == "Chunk"
                or (record.get("e.text") is not None and record.get("e.name") is None)
            )
            if is_chunk:
                properties = {
                    k: v for k, v in record.items() if k not in ["e.id", "e.text"]
                }
                text = record.get("e.text") or ""
                nodes.append(
                    ChunkNode(
                        id_=record["e.id"],
                        text=text,
                        properties=utils.remove_empty_values(properties),
                    )
                )
            else:
                properties = {
                    k: v for k, v in record.items() if k not in ["e.id", "e.name"]
                }
                name = record["e.name"] if record.get("e.name") else record["e.id"]
                label = record["e.label"] if record.get("e.label") else "Chunk"
                nodes.append(
                    EntityNode(
                        name=name,
                        label=label,
                        properties=utils.remove_empty_values(properties),
                    )
                )
        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        # Construct the Cypher query
        cypher_statement = "MATCH (e)-[r]->(t) "

        params = {}
        if entity_names or relation_names or ids:
            cypher_statement += "WHERE "

        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if relation_names and entity_names:
            cypher_statement += f"AND "
        if relation_names:
            cypher_statement += "r.label in $relation_names "
            params[f"relation_names"] = relation_names

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        # Avoid returning a massive list of triplets that represent a large portion of the graph
        # This uses the LIMIT constant defined at the top of the file
        if not (entity_names or relation_names or ids):
            return_statement = f"WHERE e.label <> 'text_chunk' RETURN * LIMIT {LIMIT};"
        else:
            return_statement = f"AND e.label <> 'text_chunk' RETURN * LIMIT {LIMIT};"

        cypher_statement += return_statement

        result = self.structured_query(cypher_statement, param_map=params)
        result = result if result else []

        # Ladybug's internal node/rel metadata keys are returned with varying case
        # depending on the query path (_LABEL/_ID/_SRC/_DST uppercase via structured_query,
        # lowercase elsewhere — see get_rel_map / base.py). Read them case-tolerantly with
        # .get() so a missing/differently-cased key skips the row instead of raising
        # KeyError: '_label'.
        def _meta(d: dict, name: str):
            return d.get(f"_{name.upper()}", d.get(f"_{name.lower()}"))

        triples = []
        for record in result:
            e, t, r = record.get("e"), record.get("t"), record.get("r")
            if not (isinstance(e, dict) and isinstance(t, dict) and isinstance(r, dict)):
                continue

            e_label = _meta(e, "label")
            t_label = _meta(t, "label")
            if e_label == "Chunk" or t_label == "Chunk":
                continue

            e_id = _meta(e, "id") or {}
            t_id = _meta(t, "id") or {}
            src_table = e_id.get("table")
            dst_table = t_id.get("table")
            id_map = {src_table: e.get("id"), dst_table: t.get("id")}
            source = EntityNode(
                name=e.get("name", e.get("id")),
                label=e_label or "Entity",
                properties=utils.get_filtered_props(e, ["_id", "_ID", "_label", "_LABEL"]),
            )
            target = EntityNode(
                name=t.get("name", t.get("id")),
                label=t_label or "Entity",
                properties=utils.get_filtered_props(t, ["_id", "_ID", "_label", "_LABEL"]),
            )
            r_src = _meta(r, "src") or {}
            r_dst = _meta(r, "dst") or {}
            rel = Relation(
                source_id=id_map.get(r_src.get("table"), "unknown"),
                target_id=id_map.get(r_dst.get("table"), "unknown"),
                label=r.get("label") or _meta(r, "label") or "",
            )
            triples.append([source, rel, target])
        return triples

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        triples = []

        ids = [node.id for node in graph_nodes]
        # Collect any triplet_source_id already set on the input nodes (by vector_query).
        # Keys are node.id = node.name for EntityNodes (LlamaIndex design).
        input_node_chunk: dict = {
            node.id: node.properties.get("triplet_source_id")
            for node in graph_nodes
            if node.properties.get("triplet_source_id")
        }
        logger.debug(
            f"Ladybug get_rel_map: {len(ids)} nodes, depth={depth}, limit={limit}, ids={ids[:5]}"
        )
        if not ids:
            logger.debug("Ladybug get_rel_map: empty ids — returning no triplets")
            return []

        # Step 1: resolve entity UUIDs and chunk mapping.
        # vector_query now returns EntityNodes (name=entity name, id=entity name).
        # We look up entities by name to get their UUIDs and associated chunk IDs.
        # The pivot also handles the legacy path where Chunk UUIDs are passed in.
        # Use connection.execute directly to bypass value_sanitize() which silently
        # drops lists of >= 128 rows in structured_query().
        try:
            _piv_result = self.connection.execute(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e)
                WHERE c.id IN $ids OR e.name IN $ids
                RETURN e.id AS entity_uuid, e.name AS entity_name, c.id AS chunk_id, label(e) AS entity_label
                LIMIT 512
                """,
                parameters={"ids": ids},
            )
            _piv_cols = _piv_result.get_column_names()
            pivot_response = [dict(zip(_piv_cols, row)) for row in _piv_result]
        except Exception as _pe:
            logger.warning("Ladybug get_rel_map: pivot query failed: %s", _pe)
            pivot_response = []
        # entity_name -> chunk_id (for triplet_source_id injection)
        entity_to_chunk: dict = dict(input_node_chunk)
        entity_uuids: list = []
        for r in pivot_response:
            ename = r.get("entity_name")
            euuid = r.get("entity_uuid")
            cid = r.get("chunk_id")
            if ename and cid and ename not in entity_to_chunk:
                entity_to_chunk[ename] = cid
            if euuid:
                entity_uuids.append(euuid)
        entity_uuids = list(set(entity_uuids))

        _used_chunk_fallback = False
        if not entity_uuids:
            # pivot returned nothing — the ids passed in are chunk UUIDs (vector_query
            # fallback path).  Resolve entity UUIDs by following MENTIONS directly from
            # those chunks, then proceed with the normal entity-to-entity traversal.
            _used_chunk_fallback = True
            try:
                # Use connection.execute directly to bypass value_sanitize() which
                # silently drops lists of >= 128 rows in structured_query().
                _cm_result = self.connection.execute(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    WHERE c.id IN $chunk_ids
                    RETURN e.id AS entity_uuid, e.name AS entity_name, c.id AS chunk_id
                    LIMIT 512
                    """,
                    parameters={"chunk_ids": ids},
                )
                _cm_cols = _cm_result.get_column_names()
                chunk_mentions = [dict(zip(_cm_cols, row)) for row in _cm_result]
                chunk_mentions = chunk_mentions or []
                for r in chunk_mentions:
                    ename = r.get("entity_name")
                    euuid = r.get("entity_uuid")
                    cid = r.get("chunk_id")
                    if ename and cid and ename not in entity_to_chunk:
                        entity_to_chunk[ename] = cid
                    if euuid:
                        entity_uuids.append(euuid)
                entity_uuids = list(set(entity_uuids))
                logger.debug(
                    "Ladybug get_rel_map: chunk fallback resolved %d chunk ids -> %d entity uuids via MENTIONS",
                    len(ids), len(entity_uuids),
                )
            except Exception as _me:
                logger.warning("Ladybug get_rel_map: chunk->entity MENTIONS lookup failed: %s", _me)

        if not entity_uuids:
            logger.debug("Ladybug get_rel_map: no entity uuids resolved — returning no triplets")
            return []

        logger.debug(
            f"Ladybug get_rel_map: resolved {len(ids)} start ids -> {len(entity_uuids)} entity uuids"
        )
        # Step 2: traverse entity-to-entity relations.
        # Use the simplest valid Ladybug single-hop pattern — filter Chunk targets
        # and MENTIONS in Python after the fact to avoid parser issues.
        # Use connection.execute directly to bypass value_sanitize() in structured_query()
        # which silently drops lists of >= 128 rows.
        try:
            _trav_result = self.connection.execute(
                f"""
                MATCH (e)-[rel]->(other)
                WHERE e.id IN $entity_uuids
                RETURN e, rel, other
                LIMIT {limit * 4};
                """,
                parameters={"entity_uuids": entity_uuids},
            )
            _trav_cols = _trav_result.get_column_names()
            response = [dict(zip(_trav_cols, row)) for row in _trav_result]
        except Exception as _qe:
            logger.warning("Ladybug get_rel_map: traversal query failed: %s", _qe)
            response = []
        logger.debug(
            f"Ladybug get_rel_map: traversal returned {len(response)} raw rows"
        )

        ignore_rels = ignore_rels or []
        if response:
            logger.debug("Ladybug get_rel_map: sample raw row keys=%s rel_keys=%s", list(response[0].keys()), list((response[0].get("rel") or {}).keys()))
        _filtered_no_rel = 0
        _filtered_mentions = 0
        _filtered_chunk_target = 0
        _filtered_empty = 0
        for record in response:
            # Single-hop query: rel is a direct relation dict, not a _RELS list
            item = record.get("rel") or {}
            rel_label = item.get("label") or item.get("_LABEL") or item.get("_label") or ""
            # Filter MENTIONS and Chunk targets in Python (WHERE on rel properties
            # and NOT (n:Label) predicates are not reliably supported in Ladybug)
            if not rel_label:
                _filtered_no_rel += 1
                continue
            if rel_label in ignore_rels or rel_label == "MENTIONS":
                _filtered_mentions += 1
                continue
            e_node = record.get("e") or {}
            other_node = record.get("other") or {}
            if other_node.get("_LABEL") == "Chunk" or other_node.get("_label") == "Chunk":
                _filtered_chunk_target += 1
                continue
            if not e_node or not other_node:
                _filtered_empty += 1
                continue

            src_name = e_node.get("name") or e_node.get("id", "")
            src_props = utils.get_filtered_props(
                e_node, ["_ID", "name", "_LABEL"]
            )
            chunk_id = entity_to_chunk.get(src_name)
            if chunk_id:
                src_props["triplet_source_id"] = chunk_id

            source = EntityNode(
                name=src_name,
                label=e_node.get("_LABEL", "Entity"),
                properties=src_props,
            )
            target = EntityNode(
                name=other_node.get("name") or other_node.get("id", ""),
                label=other_node.get("_LABEL", "Entity"),
                properties=utils.get_filtered_props(
                    other_node, ["_ID", "name", "_LABEL"]
                ),
            )
            relation = Relation(
                source_id=e_node.get("id", ""),
                target_id=other_node.get("id", ""),
                label=rel_label,
                properties=utils.get_filtered_props(
                    item, ["_ID", "_SRC", "_DST", "_LABEL", "label"]
                ),
            )
            triples.append([source, relation, target])

        logger.debug(
            f"Ladybug get_rel_map: returned {len(triples)} triplets"
            + (
                f", sample: {triples[0][0].name} -[{triples[0][1].label}]-> {triples[0][2].name}"
                if triples else ""
            )
        )
        if not triples and response:
            logger.debug(
                "Ladybug get_rel_map: all %d rows filtered — no_rel=%d mentions=%d chunk_target=%d empty=%d",
                len(response), _filtered_no_rel, _filtered_mentions, _filtered_chunk_target, _filtered_empty,
            )
            # Log a sample of what was filtered
            for _i, _rec in enumerate(response[:3]):
                _item = _rec.get("rel") or {}
                _rl = _item.get("label") or _item.get("_LABEL") or _item.get("_label") or "(none)"
                _other = _rec.get("other") or {}
                _olabel = _other.get("_LABEL") or _other.get("_label") or "(none)"
                logger.debug("  filtered row %d: rel_label=%r other._LABEL=%r item_keys=%s", _i, _rl, _olabel, list(_item.keys())[:8])
        return triples

    def get_llama_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Fetch Chunk source nodes from Ladybug by their chunk ID.

        Called by BasePGRetriever.add_source_text() to attach original chunk
        text to triplet results. The in-memory LlamaIndex docstore is empty
        after a restart, so we query Ladybug directly — Chunk records store
        their chunk UUID in the 'id' property.
        """
        if not node_ids:
            return []
        result_nodes: List[BaseNode] = []
        for chunk_id in node_ids:
            if not chunk_id:
                continue
            try:
                rows = self.structured_query(
                    "MATCH (c:Chunk {id: $id}) RETURN c.*",
                    param_map={"id": chunk_id},
                )
                if rows:
                    row = rows[0]
                    text = row.get("c.text") or ""
                    metadata = {
                        "ref_doc_id": row.get("c.ref_doc_id") or "",
                        "file_name": row.get("c.file_name") or "",
                        "file_type": row.get("c.file_type") or "",
                        "file_path": row.get("c.file_path") or "",
                    }
                    metadata["source"] = metadata["file_name"] or metadata["file_path"]
                    node = TextNode(
                        id_=str(row.get("c.id", chunk_id)),
                        text=text,
                        metadata=metadata,
                    )
                    result_nodes.append(node)
            except Exception as e:
                logger.debug(f"get_llama_nodes: failed to fetch chunk {chunk_id}: {e}")
        logger.debug(
            f"Ladybug get_llama_nodes: requested {len(node_ids)}, "
            f"returned {len(result_nodes)} TextNodes"
        )
        return result_nodes

    async def aget_llama_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Async version — delegates to sync (Ladybug client is sync-only)."""
        return self.get_llama_nodes(node_ids)

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes and relationships from the property graph store."""
        if entity_names:
            self.structured_query(
                "MATCH (n) WHERE n.name IN $entity_names DETACH DELETE n",
                param_map={"entity_names": entity_names},
            )

        if ids:
            self.structured_query(
                "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                param_map={"ids": ids},
            )

        if relation_names:
            for rel in relation_names:
                result = utils.lookup_relation(rel, self.relationship_schema)
                if result is None:
                    result = utils.lookup_relation(rel.upper(), self.relationship_schema)
                if result is None:
                    result = ("Entity", "LINKS", "Entity")
                src, _, dst = result
                self.structured_query(
                    f"""
                    MATCH (:{src})-[r {{label: $label}}]->(:{dst})
                    DELETE r
                    """,
                    param_map={"label": rel},
                )

        if properties:
            assert isinstance(properties, dict), (
                "`properties` should be a key-value mapping."
            )
            # Special case: doc_id / ref_doc_id deletion.
            # Entity nodes carry ref_doc_id directly (stored at upsert time).
            # Fall back to MENTIONS traversal for entities ingested before this fix.
            doc_id_value = properties.get("doc_id") or properties.get("ref_doc_id")
            if doc_id_value and len(properties) == 1:
                try:
                    # Primary: delete entities that have ref_doc_id stored directly
                    self.connection.execute(
                        "MATCH (e) WHERE e.ref_doc_id = $doc_id DETACH DELETE e",
                        parameters={"doc_id": doc_id_value},
                    )
                    logger.debug(
                        "Ladybug delete: removed entities with ref_doc_id=%s",
                        doc_id_value[:16],
                    )
                    # Fallback: also sweep via MENTIONS for any older entities without ref_doc_id
                    result = self.connection.execute(
                        """
                        MATCH (c:Chunk)-[:MENTIONS]->(e)
                        WHERE c.ref_doc_id = $doc_id
                        RETURN e.id AS eid
                        LIMIT 1024
                        """,
                        parameters={"doc_id": doc_id_value},
                    )
                    col_names = result.get_column_names()
                    rows = [dict(zip(col_names, row)) for row in result]
                    entity_ids = [r["eid"] for r in rows if r.get("eid")]
                    if entity_ids:
                        self.connection.execute(
                            "MATCH (e) WHERE e.id IN $ids DETACH DELETE e",
                            parameters={"ids": entity_ids},
                        )
                        logger.debug(
                            "Ladybug delete: removed %d additional entity nodes via MENTIONS for doc_id=%s",
                            len(entity_ids), doc_id_value[:16],
                        )
                except Exception as _de:
                    logger.warning("Ladybug delete by doc_id failed: %s", _de)
            else:
                cypher = "MATCH (e) WHERE "
                prop_list = []
                params = {}
                for i, prop in enumerate(properties):
                    prop_list.append(f"e.`{prop}` = $property_{i}")
                    params[f"property_{i}"] = properties[prop]
                cypher += " AND ".join(prop_list)
                self.structured_query(cypher + " DETACH DELETE e", param_map=params)

    def get_schema(self) -> Any:
        """
        Returns a structured schema of the property graph store.

        The schema contains `node_props`, `rel_props`, and `relationships` keys and
        the associated metadata.
        Example output:
        {
            'node_props': {'Chunk': [{'property': 'id', 'type': 'STRING'},
                                    {'property': 'text', 'type': 'STRING'},
                                    {'property': 'label', 'type': 'STRING'},
                                    {'property': 'embedding', 'type': 'DOUBLE'},
                                    {'property': 'properties', 'type': 'STRING'},
                                    {'property': 'ref_doc_id', 'type': 'STRING'}],
                            'Entity': [{'property': 'id', 'type': 'STRING'},
                                    {'property': 'name', 'type': 'STRING'},
                                    {'property': 'label', 'type': 'STRING'},
                                    {'property': 'embedding', 'type': 'DOUBLE'},
                                    {'property': 'properties', 'type': 'STRING'}]},
            'rel_props': {'SOURCE': [{'property': 'label', 'type': 'STRING'}]},
            'relationships': [{'end': 'Chunk', 'start': 'Chunk', 'type': 'SOURCE'}]
        }
        """
        current_table_schema = {"node_props": {}, "rel_props": {}, "relationships": []}
        node_tables = self.connection._get_node_table_names()
        for table_name in node_tables:
            node_props = self.connection._get_node_property_names(table_name)
            current_table_schema["node_props"][table_name] = []
            for prop, attr in node_props.items():
                schema = {}
                schema["property"] = prop
                schema["type"] = attr["type"]
                current_table_schema["node_props"][table_name].append(schema)

        rel_tables = self.connection._get_rel_table_names()
        for i, table in enumerate(rel_tables):
            table_name = table["name"]
            prop_values = self.connection.execute(
                f"MATCH ()-[r:{utils.quote_id(table_name)}]->() RETURN distinct r.label AS label;"
            )
            for row in prop_values:
                rel_label = row[0]
                src, dst = rel_tables[i]["src"], rel_tables[i]["dst"]
                current_table_schema["relationships"].append(
                    {"start": src, "type": rel_label, "end": dst}
                )
                current_table_schema["rel_props"][rel_label] = []
                table_details = self.connection.execute(
                    f"CALL TABLE_INFO('{table_name}') RETURN *;"
                )
                for props in table_details:
                    rel_props = {}
                    rel_props["property"] = props[1]
                    rel_props["type"] = props[2]
                    current_table_schema["rel_props"][rel_label].append(rel_props)

        self.structured_schema = current_table_schema

        return self.structured_schema

    def get_schema_str(self) -> str:
        schema = self.get_schema()

        formatted_node_props = []
        formatted_rel_props = []

        # Format node properties
        for label, props in schema["node_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_node_props.append(f"{label} {{{props_str}}}")

        # Format relationship properties
        for type, props in schema["rel_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_rel_props.append(f"{type} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})"
            for rel in schema["relationships"]
        ]

        return "\n".join(
            [
                "Node properties:",
                "\n".join(formatted_node_props),
                "Relationship properties:",
                "\n".join(formatted_rel_props),
                "The relationships:",
                "\n".join(formatted_rels),
            ]
        )


LadybugPGStore = LadybugPropertyGraphStore
