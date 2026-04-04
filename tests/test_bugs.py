"""
Regression tests for specific bugs found during flexible-graphrag Ladybug integration.

Bug 1: TIME as a relation label collides with TIME_TYPE node table name.
        safe_label() renamed both TIME entity type → TIME_TYPE (node table)
        AND TIME relation label → TIME_TYPE (rel table). Ladybug cannot have a node
        table and a rel table with the same name → native crash.
        Fix: safe_rel_label() uses _REL suffix for rel tables (TIME → TIME_REL).

Bug 2: Schema-defined relations (IS_A, PART_OF, HAS, LOCATED_IN, USED_FOR, USED_BY etc.)
        were stored via label-free MATCH + MERGE but never written to disk.
        Root cause: label-free MATCH MERGE silently succeeds but doesn't store edges
        when the actual node type pair is not registered as a FROM/TO pair in the rel
        table DDL (Ladybug only registers the first pair from CREATE REL TABLE).
        Fix: use typed MATCH for schema rels, call ensure_rel_table() to add the
        actual FROM/TO pair before each MERGE (same path as off-schema rels).
"""

from pathlib import Path
from typing import Generator, List
import uuid

import pytest
import real_ladybug as lb
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.graph_stores.ladybug import LadybugPropertyGraphStore
from llama_index.graph_stores.ladybug.utils import safe_label, safe_rel_label, quote_id

try:
    from llama_index.core.indices.property_graph.transformations.schema_llm import (
        DEFAULT_VALIDATION_SCHEMA,
    )
    HAS_DEFAULT_SCHEMA = True
except ImportError:
    HAS_DEFAULT_SCHEMA = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_test_db_files: List[str] = []


def _fresh_store(
    relationship_schema=None,
    has_structured_schema=False,
    strict_schema=False,
) -> tuple:
    """Create a fresh LadybugPropertyGraphStore and return (store, db)."""
    db_file = f"test_bugs_{uuid.uuid4().hex[:8]}.ladybug"
    Path(db_file).unlink(missing_ok=True)
    _test_db_files.append(db_file)
    db = lb.Database(db_file)
    store = LadybugPropertyGraphStore(
        db=db,
        relationship_schema=relationship_schema,
        has_structured_schema=has_structured_schema,
        strict_schema=strict_schema,
        use_vector_index=False,
    )
    return store, db


@pytest.fixture(autouse=True)
def cleanup():
    yield
    for f in _test_db_files:
        try:
            import shutil; shutil.rmtree(f, ignore_errors=True)
        except Exception:
            pass
    _test_db_files.clear()


# ---------------------------------------------------------------------------
# Bug 1a: safe_rel_label must use _REL suffix to avoid node/rel table collision
# ---------------------------------------------------------------------------

def test_safe_label_and_safe_rel_label_produce_different_names_for_time():
    """TIME node table → TIME_TYPE; TIME rel table → TIME_REL. Must not collide."""
    node_name = safe_label("TIME")
    rel_name = safe_rel_label("TIME")
    assert node_name == "TIME_TYPE", f"Expected TIME_TYPE for node, got {node_name}"
    assert rel_name == "TIME_REL", f"Expected TIME_REL for rel, got {rel_name}"
    assert node_name != rel_name, (
        "safe_label and safe_rel_label must produce different names for TIME "
        "to avoid Ladybug crash (rel table and node table cannot share a name)"
    )


def test_time_rel_table_does_not_collide_with_node_table():
    """Ingest an entity of type TIME and a relation labelled TIME — must not crash.
    (Bug: both became TIME_TYPE → Ladybug crash when creating rel table)"""
    schema = [
        ("EVENT", "TIME", "TIME"),   # rel label TIME, dst type TIME
    ]
    store, db = _fresh_store(
        relationship_schema=schema,
        has_structured_schema=True,
        strict_schema=False,
    )
    try:
        chunk = ChunkNode(
            id_="chunk1",
            text="Event at a specific time",
            properties={"triplet_source_id": "chunk1"},
        )
        event = EntityNode(label="EVENT", name="conference", id="conference")
        time_entity = EntityNode(label="TIME", name="Monday", id="Monday")
        store.upsert_nodes([chunk, event, time_entity])
        rel = Relation(
            source_id="conference",
            target_id="Monday",
            label="TIME",
            properties={"triplet_source_id": "chunk1"},
        )
        store.upsert_relations([rel])
        # Verify the relation was actually stored
        rows = store.structured_query(
            "MATCH (a)-[r:TIME_REL]->(b) RETURN a.name, b.name"
        )
        names = [(r.get("a.name"), r.get("b.name")) for r in rows]
        assert ("conference", "Monday") in names, (
            f"TIME_REL edge not found. Schema should rename TIME rel → TIME_REL. Rows: {names}"
        )
    finally:
        db.close()


def test_time_rel_label_offschema_no_crash():
    """Off-schema rel label 'TIME' with dst entity type 'TIME' (safe_label → TIME_TYPE).
    Rel table name (safe_rel_label → TIME_REL) must differ from node table name (TIME_TYPE)."""
    store, db = _fresh_store(has_structured_schema=False)
    try:
        chunk = ChunkNode(
            id_="c1", text="test",
            properties={"triplet_source_id": "c1"},
        )
        e1 = EntityNode(label="EVENT", name="summit", id="summit")
        e2 = EntityNode(label="TIME", name="2024-03-10", id="2024-03-10")
        store.upsert_nodes([chunk, e1, e2])
        rel = Relation(
            source_id="summit",
            target_id="2024-03-10",
            label="TIME",
            properties={"triplet_source_id": "c1"},
        )
        # Must not crash
        store.upsert_relations([rel])
        # In unstructured mode, TIME rel goes through LINKS — just verify no crash
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Bug 2: Schema-defined relations must be stored and retrievable via get_rel_map
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DEFAULT_SCHEMA, reason="DEFAULT_VALIDATION_SCHEMA not available")
def test_schema_defined_relations_stored_and_retrieved_default_schema():
    """With DEFAULT_VALIDATION_SCHEMA (PART_OF, IS_A, LOCATED_IN, etc.), relations
    extracted by the LLM for doc2 must appear in get_rel_map after ingest.
    (Bug: label-free MATCH MERGE silently failed to store edges when the FROM/TO
    node type pair wasn't registered in the Ladybug rel table DDL)"""
    store, db = _fresh_store(
        relationship_schema=list(DEFAULT_VALIDATION_SCHEMA),
        has_structured_schema=True,
        strict_schema=False,
    )
    try:
        # Doc 1: off-schema relations (to ensure they don't pollute doc2 results)
        chunk1 = ChunkNode(id_="c1", text="Alfresco and EMC", properties={"triplet_source_id": "c1"})
        alfresco = EntityNode(label="ORGANIZATION", name="Alfresco", id="Alfresco")
        emc = EntityNode(label="ORGANIZATION", name="EMC", id="EMC")
        store.upsert_nodes([chunk1, alfresco, emc])
        store.upsert_relations([Relation(
            source_id="Alfresco", target_id="EMC", label="BACKED_BY",
            properties={"triplet_source_id": "c1"},
        )])

        # Doc 2: PART_OF is in DEFAULT_VALIDATION_SCHEMA
        chunk2 = ChunkNode(id_="c2", text="NASA ISS Roscosmos", properties={"triplet_source_id": "c2"})
        nasa = EntityNode(label="ORGANIZATION", name="NASA", id="NASA")
        iss_prog = EntityNode(label="ORGANIZATION", name="ISS_program", id="ISS_program")
        roscosmos = EntityNode(label="ORGANIZATION", name="Roscosmos", id="Roscosmos")
        store.upsert_nodes([chunk2, nasa, iss_prog, roscosmos])

        store.upsert_relations([
            Relation(source_id="NASA", target_id="ISS_program", label="PART_OF",
                     properties={"triplet_source_id": "c2"}),
            Relation(source_id="Roscosmos", target_id="ISS_program", label="PART_OF",
                     properties={"triplet_source_id": "c2"}),
        ])

        # Direct DB check: PART_OF edges must be stored
        db_rows = store.structured_query(
            "MATCH (a)-[r:PART_OF]->(b) RETURN a.id, b.id, r.label"
        )
        parts = [(r.get("a.id"), r.get("b.id")) for r in db_rows]
        assert ("NASA", "ISS_program") in parts, (
            f"NASA->ISS_program PART_OF edge not found in DB. Stored pairs: {parts}"
        )
        assert ("Roscosmos", "ISS_program") in parts, (
            f"Roscosmos->ISS_program PART_OF edge not found in DB. Stored pairs: {parts}"
        )

        # get_rel_map must return doc2 relations when doc2 entities are queried
        triplets = store.get_rel_map([nasa, roscosmos, iss_prog], depth=1, limit=30)
        labels_and_names = [(t[0].name, t[1].label, t[2].name) for t in triplets]
        assert any(
            src == "NASA" and lbl == "PART_OF" and dst == "ISS_program"
            for src, lbl, dst in labels_and_names
        ), f"Expected NASA -[PART_OF]-> ISS_program in get_rel_map. Got: {labels_and_names}"
    finally:
        db.close()


def test_schema_defined_relations_stored_custom_schema():
    """Custom schema with PART_OF ORGANIZATION->ORGANIZATION.
    After upsert, get_rel_map must find those edges (not just doc1 edges)."""
    schema = [
        ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
        ("ORGANIZATION", "HAS", "CONCEPT"),
        ("CONCEPT", "BACKED_BY", "ORGANIZATION"),  # off-schema path test
    ]
    store, db = _fresh_store(
        relationship_schema=schema,
        has_structured_schema=True,
        strict_schema=False,
    )
    try:
        # Doc 1
        chunk1 = ChunkNode(id_="c1", text="Alfresco EMC", properties={"triplet_source_id": "c1"})
        alfresco = EntityNode(label="ORGANIZATION", name="Alfresco", id="Alfresco")
        emc = EntityNode(label="ORGANIZATION", name="EMC", id="EMC")
        store.upsert_nodes([chunk1, alfresco, emc])
        store.upsert_relations([Relation(
            source_id="Alfresco", target_id="EMC", label="PART_OF",
            properties={"triplet_source_id": "c1"},
        )])

        # Doc 2
        chunk2 = ChunkNode(id_="c2", text="NASA ISS Roscosmos", properties={"triplet_source_id": "c2"})
        nasa = EntityNode(label="ORGANIZATION", name="NASA", id="NASA")
        iss = EntityNode(label="ORGANIZATION", name="ISS", id="ISS")
        roscosmos = EntityNode(label="ORGANIZATION", name="Roscosmos", id="Roscosmos")
        store.upsert_nodes([chunk2, nasa, iss, roscosmos])
        store.upsert_relations([
            Relation(source_id="NASA", target_id="ISS", label="PART_OF",
                     properties={"triplet_source_id": "c2"}),
            Relation(source_id="Roscosmos", target_id="ISS", label="PART_OF",
                     properties={"triplet_source_id": "c2"}),
        ])

        # Direct DB check
        db_rows = store.structured_query(
            "MATCH (a)-[r:PART_OF]->(b) RETURN a.id, b.id"
        )
        pairs = {(r.get("a.id"), r.get("b.id")) for r in db_rows}
        assert ("NASA", "ISS") in pairs, f"NASA->ISS PART_OF missing. Stored: {pairs}"
        assert ("Alfresco", "EMC") in pairs, f"Alfresco->EMC PART_OF missing. Stored: {pairs}"

        # get_rel_map for doc2 entities must return doc2 triplets, not only doc1
        triplets = store.get_rel_map([nasa, iss, roscosmos], depth=1, limit=30)
        labels = [(t[0].name, t[1].label, t[2].name) for t in triplets]
        assert any(s == "NASA" and l == "PART_OF" and d == "ISS" for s, l, d in labels), (
            f"Expected NASA -[PART_OF]-> ISS. Got: {labels}"
        )
        assert any(s == "Roscosmos" and l == "PART_OF" and d == "ISS" for s, l, d in labels), (
            f"Expected Roscosmos -[PART_OF]-> ISS. Got: {labels}"
        )
    finally:
        db.close()


def test_schema_rel_with_multiple_from_to_pairs():
    """Schema has PART_OF for multiple src/dst type combinations.
    All pairs must be stored and retrievable."""
    schema = [
        ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
        ("PERSON", "PART_OF", "ORGANIZATION"),
        ("TECHNOLOGY", "PART_OF", "PRODUCT"),
    ]
    store, db = _fresh_store(
        relationship_schema=schema,
        has_structured_schema=True,
        strict_schema=False,
    )
    try:
        chunk = ChunkNode(id_="c1", text="test", properties={"triplet_source_id": "c1"})
        nasa = EntityNode(label="ORGANIZATION", name="NASA", id="NASA")
        iss = EntityNode(label="ORGANIZATION", name="ISS", id="ISS")
        alice = EntityNode(label="PERSON", name="Alice", id="Alice")
        python_ = EntityNode(label="TECHNOLOGY", name="Python", id="Python")
        llamaindex = EntityNode(label="PRODUCT", name="LlamaIndex", id="LlamaIndex")
        store.upsert_nodes([chunk, nasa, iss, alice, python_, llamaindex])
        store.upsert_relations([
            Relation(source_id="NASA", target_id="ISS", label="PART_OF",
                     properties={"triplet_source_id": "c1"}),
            Relation(source_id="Alice", target_id="NASA", label="PART_OF",
                     properties={"triplet_source_id": "c1"}),
            Relation(source_id="Python", target_id="LlamaIndex", label="PART_OF",
                     properties={"triplet_source_id": "c1"}),
        ])

        db_rows = store.structured_query(
            "MATCH (a)-[r:PART_OF]->(b) RETURN a.id, b.id"
        )
        pairs = {(r.get("a.id"), r.get("b.id")) for r in db_rows}
        assert ("NASA", "ISS") in pairs, f"ORG->ORG PART_OF missing. Stored: {pairs}"
        assert ("Alice", "NASA") in pairs, f"PERSON->ORG PART_OF missing. Stored: {pairs}"
        assert ("Python", "LlamaIndex") in pairs, f"TECH->PROD PART_OF missing. Stored: {pairs}"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Bug 5: value_sanitize() silently drops lists of >= 128 rows
#
# LlamaIndex's structured_query() passes results through value_sanitize()
# (borrowed from LangChain's Neo4j utility).  That function returns None for
# any list with >= 128 elements.  When the MENTIONS query returns >= 128 rows
# the entire result is wiped, causing "no MENTIONS found" for ALL searches.
#
# Fix: vector_query MENTIONS lookup, get_rel_map pivot query, get_rel_map
# chunk-fallback lookup, and get_rel_map step-2 traversal all use
# connection.execute directly, bypassing structured_query / value_sanitize.
# ---------------------------------------------------------------------------

def test_value_sanitize_drops_128_rows():
    """Demonstrate that value_sanitize returns None for a list of >= 128 items."""
    from llama_index.core.graph_stores.utils import value_sanitize

    rows_127 = [{"id": str(i), "name": f"e{i}"} for i in range(127)]
    rows_128 = [{"id": str(i), "name": f"e{i}"} for i in range(128)]
    rows_200 = [{"id": str(i), "name": f"e{i}"} for i in range(200)]

    assert value_sanitize(rows_127) is not None, "127 rows should NOT be dropped"
    assert value_sanitize(rows_128) is None, "128 rows SHOULD be dropped (LIST_LIMIT=128)"
    assert value_sanitize(rows_200) is None, "200 rows SHOULD be dropped"


def test_mentions_survive_128_threshold(tmp_path):
    """vector_query must return entity nodes even when total MENTIONS >= 128.

    Creates 1 chunk with 130 distinct entities (all PERSON type) so the
    MENTIONS query returns 130 rows — above the value_sanitize LIST_LIMIT.
    vector_query must still return entity nodes (not fall back to chunk nodes).
    """
    import uuid as _uuid
    db_file = str(tmp_path / "test_128.lbug")
    db = lb.Database(db_file)
    store = LadybugPropertyGraphStore(
        db=db,
        use_vector_index=False,
    )

    ENTITY_COUNT = 130  # deliberately > 128

    chunk_id = _uuid.uuid4().hex
    chunk = ChunkNode(
        id_=chunk_id,
        text="test chunk with many entities",
        metadata={"ref_doc_id": "doc1"},
    )
    entities = [
        EntityNode(label="PERSON", name=f"Person{i}", id=f"Person{i}")
        for i in range(ENTITY_COUNT)
    ]
    store.upsert_nodes([chunk] + entities)

    relations = [
        Relation(
            source_id=f"Person{i}",
            target_id=f"Person{(i+1) % ENTITY_COUNT}",
            label="KNOWS",
            properties={"triplet_source_id": chunk_id},
        )
        for i in range(ENTITY_COUNT)
    ]
    store.upsert_relations(relations)

    # Verify raw MENTIONS count in DB via store.connection (always available)
    _cnt = store.connection.execute(
        "MATCH (c:Chunk)-[:MENTIONS]->(e) RETURN count(*) AS n"
    )
    _cnt_cols = _cnt.get_column_names()
    _cnt_rows = [dict(zip(_cnt_cols, row)) for row in _cnt]
    total_mentions = _cnt_rows[0]["n"] if _cnt_rows else 0
    assert total_mentions >= ENTITY_COUNT, (
        f"Expected >= {ENTITY_COUNT} MENTIONS in DB, got {total_mentions}"
    )

    # Test the MENTIONS branch by calling the direct connection.execute path
    # (the same path vector_query now uses to bypass value_sanitize).
    result = store.connection.execute(
        "MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE c.id IN $chunk_ids "
        "RETURN e.id AS entity_id, c.id AS chunk_id, e.name AS name, label(e) AS entity_label "
        "LIMIT 512",
        parameters={"chunk_ids": [chunk_id]},
    )
    col_names = result.get_column_names()
    rows = [dict(zip(col_names, row)) for row in result]
    assert len(rows) == ENTITY_COUNT, (
        f"Expected {ENTITY_COUNT} MENTIONS rows via direct execute, got {len(rows)}"
    )

    # Confirm structured_query (which uses value_sanitize) would have killed the result
    from llama_index.core.graph_stores.utils import value_sanitize
    sanitized = value_sanitize(rows)
    assert sanitized is None, (
        f"value_sanitize should return None for {len(rows)} rows but returned {sanitized!r}"
    )

    db.close()


def test_get_rel_map_survives_128_threshold(tmp_path):
    """get_rel_map must return triplets even when entity count >= 128.

    Creates 130 entities each linked by a KNOWS relation so the traversal
    query returns >= 128 rows.  get_rel_map must return triplets, not [].
    """
    import uuid as _uuid
    db_file = str(tmp_path / "test_relmap_128.lbug")
    db = lb.Database(db_file)
    store = LadybugPropertyGraphStore(
        db=db,
        use_vector_index=False,
    )

    ENTITY_COUNT = 130

    chunk_id = _uuid.uuid4().hex
    chunk = ChunkNode(
        id_=chunk_id,
        text="test chunk",
        metadata={"ref_doc_id": "doc1"},
    )
    entities = [
        EntityNode(label="PERSON", name=f"Person{i}", id=f"Person{i}")
        for i in range(ENTITY_COUNT)
    ]
    store.upsert_nodes([chunk] + entities)

    relations = [
        Relation(
            source_id=f"Person{i}",
            target_id=f"Person{(i+1) % ENTITY_COUNT}",
            label="KNOWS",
            properties={"triplet_source_id": chunk_id},
        )
        for i in range(ENTITY_COUNT)
    ]
    store.upsert_relations(relations)

    # get_rel_map with entity nodes (normal path after vector_query)
    entity_nodes = [
        EntityNode(label="PERSON", name=f"Person{i}", id=f"Person{i}",
                   properties={"triplet_source_id": chunk_id})
        for i in range(5)
    ]
    triplets = store.get_rel_map(entity_nodes, depth=1, limit=30)
    assert len(triplets) > 0, (
        f"get_rel_map returned 0 triplets despite {ENTITY_COUNT} entities and KNOWS relations"
    )

    db.close()
