from typing import List, Optional, Tuple, _LiteralGenericAlias, get_args

import ladybug as lb

Triple = Tuple[str, str, str]


def quote_id(name: str) -> str:
    """Backtick-quote a Ladybug/Cypher identifier to avoid clashes with reserved keywords.

    Any LLM-extracted relation or entity label (e.g. IN, SET, FROM, MATCH) could be a
    reserved word.  Wrapping in backticks makes it a quoted identifier so the parser
    treats it as a name rather than a keyword.
    Backticks inside the name itself are escaped by doubling them.
    """
    return f"`{name.replace('`', '``')}`"


# Ladybug built-in type names that cause native crashes when used as node/rel
# table names even when backtick-quoted. Append _TYPE suffix to disambiguate.
_RESERVED_TYPE_NAMES = frozenset({
    "TIME", "DATE", "TIMESTAMP", "INTERVAL", "INTEGER", "BIGINT", "FLOAT",
    "DOUBLE", "BOOLEAN", "BLOB", "UUID", "JSON", "LIST", "MAP", "STRUCT",
    "UNION", "ENUM", "BIT", "HUGEINT", "DECIMAL", "VARCHAR", "TEXT",
    "TINYINT", "SMALLINT", "INT", "REAL", "CHAR", "BINARY",
})


def safe_label(name: str, existing_rel_tables: Optional[List[str]] = None) -> str:
    """Return a safe node table name.

    Appends _TYPE for Ladybug built-in type names that crash when used as node
    table names even when backtick-quoted.

    Also checks existing_rel_tables: if the LLM extracts an entity label that
    matches an already-created relation table (e.g. ROLE was created as a rel
    table by a prior ingest), appends _TYPE to avoid a Binder exception when
    the entity MERGE query is executed.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)
    upper = name.upper()
    if upper in _RESERVED_TYPE_NAMES:
        return name + "_TYPE"
    if existing_rel_tables:
        rel_table_upper = {t.upper() if isinstance(t, str) else t.get("name", "").upper() for t in existing_rel_tables}
        if upper in rel_table_upper:
            _log.debug("Ladybug safe_label: '%s' collides with existing rel table — using '%s_TYPE'", name, name)
            return name + "_TYPE"
    return name


def safe_rel_label(name: str, existing_node_tables: Optional[List[str]] = None) -> str:
    """Return a safe relationship table name for a given relation label.

    Uses _REL suffix (not _TYPE) to avoid collision with node tables:
    a rel table and a node table cannot share the same name in Ladybug.
    For example, if the LLM extracts a relation labelled 'TIME', the node
    table for TIME entities is 'TIME_TYPE', so the rel table must be 'TIME_REL'.

    Also checks existing_node_tables: if the LLM extracts a relation label that
    matches a dynamically-created node table (e.g. FEATURE), appends _REL to
    avoid a native crash when CREATE REL TABLE is called with that name.
    """
    upper = name.upper()
    if upper in _RESERVED_TYPE_NAMES:
        return name + "_REL"
    if existing_node_tables:
        node_table_upper = {t.upper() for t in existing_node_tables}
        if upper in node_table_upper:
            return name + "_REL"
    return name


def create_fresh_database(db: str) -> None:
    """
    Create a new Ladybug database by removing existing database directory and its contents.
    """
    import shutil

    shutil.rmtree(db, ignore_errors=True)


def get_list_from_literal(literal: _LiteralGenericAlias) -> List[str]:
    """
    Get a list of strings from a Literal type.

    Parameters
    ----------
    literal (_LiteralGenericAlias): The Literal type from which to extract the strings.

    Returns
    -------
    List[str]: A list of strings extracted from the Literal type.

    """
    if not isinstance(literal, _LiteralGenericAlias):
        raise TypeError(
            f"{literal} must be a Literal type.\nTry using typing.Literal{literal}."
        )
    return list(get_args(literal))


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters
    ----------
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns
    -------
    dict: A new dictionary with all empty values removed.

    """
    # Create a new dictionary excluding empty values and remove the `e.` prefix from the keys
    return {key.replace("e.", ""): value for key, value in input_dict.items() if value}


def get_filtered_props(records: dict, filter_list: List[str]) -> dict:
    return {k: v for k, v in records.items() if k not in filter_list}


# Lookup entry by middle value of tuple
def lookup_relation(relation: str, triples: List[Triple]) -> Triple:
    """
    Look up a triple in a list of triples by the middle value.
    Tries exact match first, then case-insensitive match.
    Returns None if no match found.
    """
    # Exact match first
    for triple in triples:
        if triple[1] == relation:
            return triple
    # Case-insensitive fallback
    relation_upper = relation.upper()
    for triple in triples:
        if triple[1].upper() == relation_upper:
            return triple
    return None


def create_chunk_node_table(
    connection: lb.Connection, embedding_dimension: Optional[int] = None
) -> None:
    # For now, the additional `properties` dict from LlamaIndex is stored as a string
    # TODO: See if it makes sense to add better support for property metadata as columns

    embedding_type = (
        f"DOUBLE[{embedding_dimension}]" if embedding_dimension else "DOUBLE[]"
    )

    connection.execute(
        f"""
        CREATE NODE TABLE IF NOT EXISTS Chunk (
            id STRING,
            text STRING,
            label STRING,
            embedding {embedding_type},
            creation_date DATE,
            last_modified_date DATE,
            file_name STRING,
            file_path STRING,
            file_size INT64,
            file_type STRING,
            ref_doc_id STRING,
            PRIMARY KEY(id)
        )
        """
    )


def create_entity_node_tables(connection: lb.Connection, entities: List[str]) -> None:
    for tbl_name in entities:
        # Entity tables don't need embedding columns - only Chunk nodes have embeddings
        # For now, the additional `properties` dict from LlamaIndex is stored as a string
        # TODO: See if it makes sense to add better support for property metadata as columns
        connection.execute(
            f"""
            CREATE NODE TABLE IF NOT EXISTS {quote_id(tbl_name)} (
                id STRING,
                name STRING,
                label STRING,
                creation_date DATE,
                last_modified_date DATE,
                file_name STRING,
                file_path STRING,
                file_size INT64,
                file_type STRING,
                triplet_source_id STRING,
                ref_doc_id STRING,
                PRIMARY KEY(id)
            )
            """
        )


def create_entity_relationship_table(
    connection: lb.Connection, label: str, src_id: str, dst_id: str
) -> None:
    connection.execute(
        f"""
        CREATE REL TABLE IF NOT EXISTS {quote_id(label)} (
            FROM {quote_id(src_id)} TO {quote_id(dst_id)},
            label STRING,
            triplet_source_id STRING
        );
        """
    )


def create_catch_all_links_table(
    connection: lb.Connection, all_entity_types: List[str]
) -> None:
    """Create a LINKS rel table for Entity->Entity catch-all and MENTIONS for all types.
    Individual LINKS type-pair entries are added on demand via ensure_links_pair().
    """
    connection.execute(
        "CREATE REL TABLE IF NOT EXISTS LINKS (FROM Entity TO Entity, label STRING, triplet_source_id STRING)"
    )
    # Ensure Entity is a valid MENTIONS target (needed when off-schema nodes fall back
    # to Entity table but MENTIONS was created before Entity table existed).
    # Also creates MENTIONS if it doesn't exist yet (unstructured mode).
    try:
        connection.execute(
            "CREATE REL TABLE IF NOT EXISTS MENTIONS "
            "(FROM Chunk TO Entity, label STRING, triplet_source_id STRING)"
        )
    except Exception:
        # MENTIONS already exists — just add the Chunk->Entity pair
        try:
            connection.execute("ALTER TABLE MENTIONS ADD FROM Chunk TO Entity")
        except Exception:
            pass  # Already exists


def ensure_links_pair(
    connection: lb.Connection, src_type: str, dst_type: str
) -> None:
    """Add a FROM src_type TO dst_type pair to LINKS if not already present."""
    connection.execute(
        f"ALTER TABLE LINKS ADD IF NOT EXISTS FROM {quote_id(src_type)} TO {quote_id(dst_type)}"
    )


def ensure_rel_table(
    connection: lb.Connection, rel_label: str, src_type: str, dst_type: str
) -> str:
    """Ensure a named rel table exists for rel_label with the given FROM/TO pair.
    Used in non-strict mode to create typed rel tables on the fly for off-schema
    relation labels the LLM extracted.

    Returns the actual table name to use (always rel_label — no variant tables needed).

    Uses CREATE REL TABLE IF NOT EXISTS for first-time creation, then
    ALTER TABLE ... ADD IF NOT EXISTS FROM ... TO ... for additional type pairs.
    Both are idempotent so this is safe to call repeatedly for the same pair.
    """
    connection.execute(
        f"CREATE REL TABLE IF NOT EXISTS {quote_id(rel_label)} "
        f"(FROM {quote_id(src_type)} TO {quote_id(dst_type)}, label STRING, triplet_source_id STRING)"
    )
    connection.execute(
        f"ALTER TABLE {quote_id(rel_label)} ADD IF NOT EXISTS FROM {quote_id(src_type)} TO {quote_id(dst_type)}"
    )
    return rel_label


def create_relation_tables(
    connection: lb.Connection, entities: List[str], relationship_schema: List[Triple]
) -> None:
    # Group all (src, dst) pairs by rel_label so we can:
    #   1. CREATE REL TABLE IF NOT EXISTS on the first pair
    #   2. ALTER TABLE ... ADD IF NOT EXISTS for all subsequent pairs
    # This ensures every FROM/TO combination the schema defines is registered,
    # so typed MATCH MERGE works correctly for all entity type pairs.
    from collections import defaultdict
    pairs_by_label: dict = defaultdict(list)
    for src, rel_label, dst in relationship_schema:
        pairs_by_label[rel_label].append((src, dst))

    for rel_label, pairs in pairs_by_label.items():
        if not pairs:
            continue
        # Create the table with the first pair
        first_src, first_dst = pairs[0]
        connection.execute(
            f"""
            CREATE REL TABLE IF NOT EXISTS {quote_id(rel_label)} (
                FROM {quote_id(first_src)} TO {quote_id(first_dst)},
                label STRING,
                triplet_source_id STRING
            );
            """
        )
        # Register all remaining pairs via ALTER TABLE
        for src, dst in pairs[1:]:
            try:
                connection.execute(
                    f"ALTER TABLE {quote_id(rel_label)} ADD IF NOT EXISTS FROM {quote_id(src)} TO {quote_id(dst)}"
                )
            except Exception:
                pass  # Pair already registered

    # Build MENTIONS rel table: FROM Chunk TO each non-Chunk entity type
    target_types = list(set(e for e in entities if e != "Chunk"))
    if not target_types:
        return
    ddl = "CREATE REL TABLE IF NOT EXISTS MENTIONS ("
    ddl += ", ".join(f"FROM Chunk TO {t}" for t in target_types)
    ddl += ", label STRING, triplet_source_id STRING)"
    connection.execute(ddl)
