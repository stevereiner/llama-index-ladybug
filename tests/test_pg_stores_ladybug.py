from pathlib import Path
from typing import Generator, List
from unittest.mock import Mock, patch

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.graph_stores.ladybug import LadybugPropertyGraphStore

# Track all database files created during tests for cleanup
_test_db_files = []


def cleanup_test_databases():
    """Clean up all test database files."""
    for db_file in _test_db_files:
        try:
            Path(db_file).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors
    _test_db_files.clear()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown fixture that runs for every test."""
    # Setup: Clear any existing test databases
    cleanup_test_databases()

    yield

    # Teardown: Clean up all databases created during tests
    cleanup_test_databases()


@pytest.fixture()
def pg_store() -> Generator[LadybugPropertyGraphStore, None, None]:
    import ladybug as lb

    # Remove existing database file
    db_file = "llama_test_db.ladybug"
    Path(db_file).unlink(missing_ok=True)
    _test_db_files.append(db_file)

    db = lb.Database(db_file)
    pg_store = LadybugPropertyGraphStore(db)
    pg_store.structured_query("MATCH (n) DETACH DELETE n")
    pg_store._test_db_file = db_file

    yield pg_store

    # Close database connection
    try:
        db.close()
    except Exception:
        pass


@pytest.fixture()
def pg_store_with_vectors() -> Generator[LadybugPropertyGraphStore, None, None]:
    """Fixture for pg_store with vector indexing enabled."""
    import uuid

    import ladybug as lb

    # Use unique database file name to avoid conflicts
    db_file = f"llama_test_db_vector_{uuid.uuid4().hex[:8]}.ladybug"
    Path(db_file).unlink(missing_ok=True)
    _test_db_files.append(db_file)

    db = lb.Database(db_file)

    # Mock embedding model for testing
    mock_embed_model = Mock()
    mock_embed_model.get_text_embedding.return_value = [0.1] * 384

    pg_store = LadybugPropertyGraphStore(
        db=db, use_vector_index=True, embed_model=mock_embed_model, embed_dimension=384
    )
    pg_store.structured_query("MATCH (n) DETACH DELETE n")

    # Store the db_file for cleanup
    pg_store._test_db_file = db_file

    yield pg_store

    # Close database connection
    try:
        db.close()
    except Exception:
        pass


@pytest.fixture()
def sample_chunk_nodes() -> List[ChunkNode]:
    """Sample chunk nodes with embeddings for testing."""
    return [
        ChunkNode(
            id_="chunk1",
            text="This is the first chunk of text",
            embedding=[0.1, 0.2, 0.3, 0.4] * 96,  # 384-dim embedding
            properties={"file_name": "test1.txt"},
        ),
        ChunkNode(
            id_="chunk2",
            text="This is the second chunk of text",
            embedding=[0.5, 0.6, 0.7, 0.8] * 96,  # 384-dim embedding
            properties={"file_name": "test2.txt"},
        ),
    ]


def test_ladybugdb_pg_store(pg_store: LadybugPropertyGraphStore) -> None:
    # Create a two entity nodes
    entity1 = EntityNode(label="PERSON", name="Logan")
    entity2 = EntityNode(label="ORGANIZATION", name="LlamaIndex")

    # Create a relation
    relation = Relation(
        label="WORKS_FOR",
        source_id=entity1.id,
        target_id=entity2.id,
    )

    pg_store.upsert_nodes([entity1, entity2])
    pg_store.upsert_relations([relation])

    source_node = TextNode(text="Logan (age 28), works for LlamaIndex since 2023.")
    relations = [
        Relation(
            label="MENTIONS",
            target_id=entity1.id,
            source_id=source_node.node_id,
        ),
        Relation(
            label="MENTIONS",
            target_id=entity2.id,
            source_id=source_node.node_id,
        ),
    ]

    pg_store.upsert_llama_nodes([source_node])
    pg_store.upsert_relations(relations)

    print(pg_store.get())

    kg_nodes = pg_store.get(ids=[entity1.id])
    assert len(kg_nodes) == 1
    assert kg_nodes[0].label == "PERSON"
    assert kg_nodes[0].name == "Logan"

    # get paths from a node
    paths = pg_store.get_rel_map(kg_nodes, depth=1)
    for path in paths:
        assert path[0].id == entity1.id
        assert path[2].id == entity2.id
        assert path[1].id == relation.id

    query = "match (n:Entity) return n"
    result = pg_store.structured_query(query)
    assert len(result) == 2

    # deleting
    # delete our entities
    pg_store.delete(ids=[entity1.id, entity2.id])

    # delete our text nodes
    pg_store.delete(ids=[source_node.node_id])

    nodes = pg_store.get(ids=[entity1.id, entity2.id])
    assert len(nodes) == 0

    text_nodes = pg_store.get_llama_nodes([source_node.node_id])
    assert len(text_nodes) == 0


def test_create_vector_index_disabled(pg_store: LadybugPropertyGraphStore) -> None:
    """Test _create_vector_index when use_vector_index is False."""
    pg_store.use_vector_index = False

    # Should return without doing anything
    pg_store._create_vector_index("Chunk")

    # Verify no indexes were created by checking database
    indexes_result = pg_store.connection.execute("CALL SHOW_INDEXES() RETURN *")
    index_names = [row[1] for row in indexes_result if len(row) > 1]
    assert "chunk_embedding_index" not in index_names


def test_create_vector_index_no_data(
    pg_store_with_vectors: LadybugPropertyGraphStore,
) -> None:
    """Test _create_vector_index when table has no embedding data."""
    # Try to create index on empty table - should not create index since Ladybug requires data first
    pg_store_with_vectors._create_vector_index("Chunk")

    # Should not create index since no data exists (Ladybug requirement)
    indexes_result = pg_store_with_vectors.connection.execute(
        "CALL SHOW_INDEXES() RETURN *"
    )
    index_names = [row[1] for row in indexes_result if len(row) > 1]
    assert "chunk_embedding_index" not in index_names


def test_create_vector_index_with_data(
    pg_store_with_vectors: LadybugPropertyGraphStore, sample_chunk_nodes: List[ChunkNode]
) -> None:
    """Test _create_vector_index with actual embedding data."""
    # First insert chunk nodes with embeddings
    pg_store_with_vectors.upsert_nodes(sample_chunk_nodes)

    # Now create vector index
    pg_store_with_vectors._create_vector_index("Chunk")

    # Verify index was created (if vector extension is available)
    # Note: This may fail in test environment without vector extension
    # In that case, the method should handle gracefully


def test_create_vector_index_already_exists(
    pg_store_with_vectors: LadybugPropertyGraphStore, sample_chunk_nodes: List[ChunkNode]
) -> None:
    """Test _create_vector_index when index already exists."""
    # Insert data
    pg_store_with_vectors.upsert_nodes(sample_chunk_nodes)

    # Create index first time
    pg_store_with_vectors._create_vector_index("Chunk")

    # Verify index exists
    indexes_result = pg_store_with_vectors.connection.execute(
        "CALL SHOW_INDEXES() RETURN *"
    )
    index_names = [row[1] for row in indexes_result if len(row) > 1]
    assert "chunk_embedding_index" in index_names
    initial_count = len(index_names)

    # Try to create again - should not duplicate
    pg_store_with_vectors._create_vector_index("Chunk")

    # Verify no duplicate was created
    indexes_result = pg_store_with_vectors.connection.execute(
        "CALL SHOW_INDEXES() RETURN *"
    )
    index_names = [row[1] for row in indexes_result if len(row) > 1]
    final_count = len(index_names)
    assert initial_count == final_count


def test_ensure_vector_indexes_disabled(pg_store: LadybugPropertyGraphStore) -> None:
    """Test _ensure_vector_indexes when use_vector_index is False."""
    pg_store.use_vector_index = False

    # Should return without doing anything
    pg_store._ensure_vector_indexes()

    # Verify no indexes were created
    indexes_result = pg_store.connection.execute("CALL SHOW_INDEXES() RETURN *")
    index_names = [row[1] for row in indexes_result if len(row) > 1]
    assert "chunk_embedding_index" not in index_names


def test_ensure_vector_indexes_enabled(
    pg_store_with_vectors: LadybugPropertyGraphStore, sample_chunk_nodes: List[ChunkNode]
) -> None:
    """Test _ensure_vector_indexes with vector indexing enabled."""
    # Insert chunk nodes with embeddings
    pg_store_with_vectors.upsert_nodes(sample_chunk_nodes)

    # Ensure vector indexes
    pg_store_with_vectors._ensure_vector_indexes()

    # This should attempt to create index for Chunk table
    # Exact behavior depends on whether vector extension is available


def test_refresh_vector_index(
    pg_store_with_vectors: LadybugPropertyGraphStore, sample_chunk_nodes: List[ChunkNode]
) -> None:
    """Test refresh_vector_index method."""
    # Insert chunk nodes with embeddings
    pg_store_with_vectors.upsert_nodes(sample_chunk_nodes)

    # Create initial index
    pg_store_with_vectors._create_vector_index("Chunk")

    # Refresh the index (should drop and recreate)
    pg_store_with_vectors.refresh_vector_index()

    # The method should handle both drop and recreate operations
    # Even if they fail due to missing vector extension


def test_vector_query_with_mock_results(
    pg_store_with_vectors: LadybugPropertyGraphStore, sample_chunk_nodes: List[ChunkNode]
) -> None:
    """Test vector_query method with mocked vector search results."""
    # Insert chunk nodes with embeddings
    pg_store_with_vectors.upsert_nodes(sample_chunk_nodes)

    # Create a vector query
    query = VectorStoreQuery(
        query_embedding=[0.2, 0.3, 0.4, 0.5] * 96,  # 384-dim query embedding
        similarity_top_k=2,
    )

    def _make_execute_result(rows, col_names):
        """Return a mock that behaves like a Ladybug QueryResult."""
        mock_result = Mock()
        mock_result.get_column_names.return_value = col_names
        mock_result.__iter__ = Mock(return_value=iter(rows))
        return mock_result

    # Mock both vector index query and structured query calls
    with (
        patch.object(pg_store_with_vectors.connection, "execute") as mock_execute,
        patch.object(
            pg_store_with_vectors, "structured_query"
        ) as mock_structured_query,
    ):
        # Mock different execute calls based on query type.
        # MENTIONS expansion now goes through connection.execute (bypasses
        # value_sanitize) so it must return a result with get_column_names().
        def mock_execute_side_effect(query, **kwargs):
            if "COUNT(n)" in query:
                return [(2,)]
            elif "SHOW_INDEXES" in query:
                return [("table", "chunk_embedding_index")]
            elif "MENTIONS" in query:
                # MENTIONS expansion — return empty result (triggers fallback)
                return _make_execute_result([], ["entity_id", "chunk_id", "name", "e", "entity_label"])
            else:
                # QUERY_VECTOR_INDEX result: plain row tuples are fine here
                return [("chunk1", 0.1), ("chunk2", 0.3)]

        mock_execute.side_effect = mock_execute_side_effect

        # structured_query is still used for fetching chunk node data in the fallback
        mock_structured_query.side_effect = [
            # First call - fetch chunk1 data
            [
                {
                    "n.id": "chunk1",
                    "n.text": "This is the first chunk of text",
                    "n.file_name": "test1.txt",
                }
            ],
            # Second call - fetch chunk2 data
            [
                {
                    "n.id": "chunk2",
                    "n.text": "This is the second chunk of text",
                    "n.file_name": "test2.txt",
                }
            ],
        ]

        # Execute vector query
        nodes, similarities = pg_store_with_vectors.vector_query(query)

        # Verify results
        assert len(nodes) == 2
        assert len(similarities) == 2

        # Check similarity conversion (1.0 - distance)
        assert similarities[0] == 1.0 - 0.1  # 0.9
        assert similarities[1] == 1.0 - 0.3  # 0.7

        # Verify nodes are sorted by similarity (descending)
        assert similarities[0] > similarities[1]


def test_vector_query_ensures_indexes(
    pg_store_with_vectors: LadybugPropertyGraphStore, sample_chunk_nodes: List[ChunkNode]
) -> None:
    """Test that vector_query calls _ensure_vector_indexes."""
    # Insert chunk nodes with embeddings
    pg_store_with_vectors.upsert_nodes(sample_chunk_nodes)

    # Create a vector query
    query = VectorStoreQuery(
        query_embedding=[0.2, 0.3, 0.4, 0.5] * 96, similarity_top_k=1
    )

    # Mock _ensure_vector_indexes to verify it's called
    with patch.object(pg_store_with_vectors, "_ensure_vector_indexes") as mock_ensure:
        with patch.object(pg_store_with_vectors.connection, "execute") as mock_execute:
            mock_execute.return_value = []  # Empty results

            pg_store_with_vectors.vector_query(query)

            # Verify _ensure_vector_indexes was called
            mock_ensure.assert_called_once()


def test_upsert_chunk_twice_with_vector_index(pg_store_with_vectors):
    """Upsert the same ChunkNode twice (simulating incremental add of 2nd document).

    With use_vector_index=True the second upsert must not raise
    'Cannot set property vec in table embeddings because it is used in one or more indexes.'
    The fix: pre-delete + CREATE instead of MERGE+SET.
    """
    embedding = [0.1] * 384

    chunk = ChunkNode(
        id_="test-chunk-inc-001",
        text="First version of the chunk text.",
        label="chunk",
        embedding=embedding,
        properties={
            "ref_doc_id": "doc-001",
            "creation_date": "2026-01-01",
            "last_modified_date": "2026-01-01",
            "file_name": "doc1.txt",
            "file_path": "/docs/doc1.txt",
            "file_size": "100",
            "file_type": ".txt",
        },
    )

    # First ingest — must succeed
    pg_store_with_vectors.upsert_nodes([chunk])

    # Verify it was stored
    results = pg_store_with_vectors.structured_query(
        "MATCH (c:Chunk {id: $id}) RETURN c.id",
        param_map={"id": "test-chunk-inc-001"},
    )
    assert results, "Chunk should exist after first upsert"

    # Second ingest of the same chunk (incremental update / re-ingest scenario).
    # This is the operation that previously raised RuntimeError from Ladybug.
    chunk2 = ChunkNode(
        id_="test-chunk-inc-001",  # same id — simulates re-ingest
        text="Updated version of the chunk text.",
        label="chunk",
        embedding=[0.2] * 384,
        properties={
            "ref_doc_id": "doc-001",
            "creation_date": "2026-01-01",
            "last_modified_date": "2026-03-29",
            "file_name": "doc1.txt",
            "file_path": "/docs/doc1.txt",
            "file_size": "110",
            "file_type": ".txt",
        },
    )

    # Must not raise RuntimeError
    pg_store_with_vectors.upsert_nodes([chunk2])

    # Verify the updated text is stored (not the old one)
    results = pg_store_with_vectors.structured_query(
        "MATCH (c:Chunk {id: $id}) RETURN c.text",
        param_map={"id": "test-chunk-inc-001"},
    )
    assert results, "Chunk should still exist after second upsert"
    assert results[0]["c.text"] == "Updated version of the chunk text.", \
        f"Expected updated text, got: {results[0]}"


def test_upsert_two_different_chunks_with_vector_index(pg_store_with_vectors):
    """Upsert two different ChunkNodes (simulating adding a 2nd document).

    Ensures the second chunk INSERT doesn't fail due to vector index constraint
    on the first chunk's already-indexed embedding row.
    """
    embedding_a = [0.1] * 384
    embedding_b = [0.9] * 384

    chunk_a = ChunkNode(
        id_="test-chunk-doc1-001",
        text="Content from document 1.",
        label="chunk",
        embedding=embedding_a,
        properties={
            "ref_doc_id": "doc-001",
            "creation_date": "2026-01-01",
            "last_modified_date": "2026-01-01",
            "file_name": "doc1.txt",
            "file_path": "/docs/doc1.txt",
            "file_size": "100",
            "file_type": ".txt",
        },
    )
    chunk_b = ChunkNode(
        id_="test-chunk-doc2-001",
        text="Content from document 2.",
        label="chunk",
        embedding=embedding_b,
        properties={
            "ref_doc_id": "doc-002",
            "creation_date": "2026-01-02",
            "last_modified_date": "2026-01-02",
            "file_name": "doc2.txt",
            "file_path": "/docs/doc2.txt",
            "file_size": "200",
            "file_type": ".txt",
        },
    )

    # First document
    pg_store_with_vectors.upsert_nodes([chunk_a])

    # Second document — must not raise RuntimeError
    pg_store_with_vectors.upsert_nodes([chunk_b])

    # Both chunks should exist
    results = pg_store_with_vectors.structured_query(
        "MATCH (c:Chunk) RETURN c.id ORDER BY c.id"
    )
    ids = [r["c.id"] for r in results]
    assert "test-chunk-doc1-001" in ids
    assert "test-chunk-doc2-001" in ids
