"""Verify LadybugPropertyGraphStore.get_triplets() — regression check for flexible-graphrag issue #19.

Reported against flexible-graphrag (https://github.com/stevereiner/flexible-graphrag/issues/19),
but the bug lived here: before the fix, get_triplets() raised `KeyError: '_label'` because it
assumed lowercase internal keys, while structured_query returns them uppercase
(_LABEL/_ID/_SRC/_DST).

This is self-contained: builds a temp store, upserts a couple of entities + a relation,
then calls get_triplets() / get_entities() and prints the results.
Run:  python tests/verify_get_triplets.py
"""

import os
import tempfile

import ladybug as lb
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.graph_stores.ladybug import LadybugPropertyGraphStore


def main() -> None:
    db = lb.Database(os.path.join(tempfile.mkdtemp(), "verify_db"))
    # use_vector_index=False keeps this focused on the graph API (no VECTOR extension needed)
    store = LadybugPropertyGraphStore(db, use_vector_index=False)

    alice = EntityNode(label="PERSON", name="Alice")
    acme = EntityNode(label="ORGANIZATION", name="Acme")
    store.upsert_nodes([alice, acme])
    store.upsert_relations(
        [Relation(label="WORKS_FOR", source_id=alice.id, target_id=acme.id)]
    )

    print("get_triplets():")
    triplets = store.get_triplets()
    for src, rel, tgt in triplets:
        print(f"  ({src.name})-[{rel.label}]->({tgt.name})")
    assert triplets, "expected at least one triplet"
    assert any(rel.label == "WORKS_FOR" for _, rel, _ in triplets), "WORKS_FOR missing"

    print("\nget_triplets(entity_names=['Alice']):")
    for src, rel, tgt in store.get_triplets(entity_names=["Alice"]):
        print(f"  ({src.name})-[{rel.label}]->({tgt.name})")

    print("\nget_entities():", store.get_entities())

    print("\nOK - get_triplets() returned rows with no KeyError: '_label'")


if __name__ == "__main__":
    main()
