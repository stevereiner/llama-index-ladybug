# llama-index-ladybug

[LlamaIndex](https://www.llamaindex.ai/) graph store integration for [Ladybug](https://github.com/LadybugDB/ladybug) — an embedded graph database built for query speed and scalability. Ladybug is optimized for handling complex analytical workloads on very large databases and provides a set of retrieval features, such as full text search and vector indices.

The database was formerly known as [Kùzu](https://kuzudb.com/).

## Installation

```bash
uv pip install llama-index-graph-stores-ladybug
```

### Vector index extension (Ladybug 0.18.x+)

With `use_vector_index=True` (the default), the store loads Ladybug's **VECTOR extension** on first
use (`INSTALL vector; LOAD vector;`). `INSTALL` downloads it over the network the first time, then
caches it under `~/.lbdb/extension/` — after that it works offline. Set `use_vector_index=False` to
skip vector indexing entirely.

#### OpenSSL 3 requirement

The VECTOR extension uses **OpenSSL 3** when it is installed and loaded. OpenSSL is **not** bundled
with Ladybug, so it must be present on your system — install it yourself and keep it patched. If it's
missing, vector indexing is unavailable.

##### Windows

The extension needs `libssl-3-x64.dll` and `libcrypto-3-x64.dll`. Install with Chocolatey, use PowerShell, Command Prompt, or Terminal with **Run as administrator** (elevated):

```powershell
choco install openssl.light
```

This installs OpenSSL 3.x to `C:\Program Files\OpenSSL`, copies `libssl-3-x64.dll` /
`libcrypto-3-x64.dll` into `C:\Windows\System32`, and appends `C:\Program Files\OpenSSL\bin` to the
system `PATH`. Because the DLLs land in System32, `LOAD vector` then works in any shell with no
further `PATH` setup.

**Then verify** in a **new** shell (so `PATH` updates):

```powershell
where.exe libssl-3-x64.dll
where.exe libcrypto-3-x64.dll
```

Important notes:

- **It must be OpenSSL 3.x.** OpenSSL 4 renames the libraries (`libssl-4-x64.dll`), which won't
  work — `winget install ShiningLight.OpenSSL.Light` currently ships 4.x, so avoid it.
- Names must be **exactly** `libssl-3-x64.dll` / `libcrypto-3-x64.dll` (not `libssl-3.dll` or
  `libeay32.dll`), both from the same OpenSSL version.
- If choco says it's already installed but the files are gone, `choco uninstall openssl.light` first.
- Open a **new** shell afterwards; services, Docker, and some IDE terminals don't inherit `PATH`.

##### macOS

```bash
brew install openssl@3
```

Apple's bundled `/usr/bin/openssl` is LibreSSL, not OpenSSL, and isn't a substitute. Homebrew's
`openssl@3` is keg-only, so if the extension still can't find it, expose the Homebrew lib directory
(e.g. add `$(brew --prefix openssl@3)/lib` to `DYLD_LIBRARY_PATH`).

##### Linux

The distro provides OpenSSL 3 (`libssl.so.3` / `libcrypto.so.3`) and it's usually already installed.
If not:

| Distro | Command |
| --- | --- |
| Debian / Ubuntu | `apt install libssl3` |
| Fedora / RHEL | `dnf install openssl-libs` |
| Alpine | `apk add openssl` |

Slim container images often omit it — install it in the image if you hit a load failure.

## Quick Start

### LadybugPropertyGraphStore — unstructured (default)

No schema required. All LLM-extracted entities are stored as Entity type and only relation types are Links and Mentions.

```python
from pathlib import Path
import ladybug as lb
from llama_index.graph_stores.ladybug import LadybugPropertyGraphStore
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Create a Ladybug database
Path("my_graph.ladybug").unlink(missing_ok=True)
db = lb.Database("my_graph.ladybug")

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

graph_store = LadybugPropertyGraphStore(
    db,
    use_vector_index=True,
    embed_model=embed_model,
)

documents = SimpleDirectoryReader("./data").load_data()

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = index.as_query_engine()
response = query_engine.query("What are the main topics in these documents?")
print(response)
```

### LadybugPropertyGraphStore — structured schema

Pass a `relationship_schema` to guide the LLM towards your schema.

`strict_schema=False` (default) allows the graph to expand beyond the declared types — off-schema entities and relations are stored in overflow tables alongside the schema-defined ones.

`strict_schema=True` enforces the schema strictly — off-schema entities and relations are silently dropped at ingest.

```python
graph_store = LadybugPropertyGraphStore(
    db,
    relationship_schema=[
        ("PERSON", "WORKS_FOR", "ORGANIZATION"),
        ("PERSON", "KNOWS", "PERSON"),
    ],
    has_structured_schema=True,
    strict_schema=False,   # True to reject off-schema types entirely
    use_vector_index=True,
    embed_model=embed_model,
)
```

### LadybugGraphStore

```python
import ladybug as lb
from llama_index.graph_stores.ladybug import LadybugGraphStore
from llama_index.core import KnowledgeGraphIndex, StorageContext, SimpleDirectoryReader

db = lb.Database("my_graph.ladybug")
graph_store = LadybugGraphStore(db)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

documents = SimpleDirectoryReader("./data").load_data()

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

query_engine = index.as_query_engine()
response = query_engine.query("What are the main topics in these documents?")
print(response)
```

## Features

- **Embedded** — no server required; the database is a local directory
- **Cypher queries** — full Cypher support via `structured_query()`
- **Vector index** — HNSW vector index on chunk nodes for similarity search, built into the graph store
- **Structured schemas** — optionally enforce entity/relation types for higher-quality triple extraction
- **Both graph store APIs** — supports both `PropertyGraphIndex` (`LadybugPropertyGraphStore`) and the legacy `KnowledgeGraphIndex` (`LadybugGraphStore`)

## Documentation

- [API Reference](docs/api/README.md)
- [Property Graph Notebook](docs/notebooks/property_graph_ladybug.ipynb)
- [GraphStore Notebook](docs/notebooks/LadybugGraphDemo.ipynb)

## Development

```bash
# Clone and set up
git clone https://github.com/stevereiner/llama-index-ladybug
cd llama-index-ladybug
uv sync --group dev

# Run tests
pytest

# Install pre-commit hooks (strips notebook outputs on commit)
pre-commit install
```

## Acknowledgements

Started from the Kuzu → Ladybug llama-index support port by [@adsharma](https://github.com/adsharma) ([PR #20232](https://github.com/run-llama/llama_index/pull/20232)) — a proposed LadybugDB (formerly Kùzu) integration into the upstream llama-index repo.

## Requirements

- Python 3.10+
- `ladybug >= 0.18.2`
- `llama-index-core >= 0.14.20`
- For the vector index on Ladybug 0.18.x+: the downloadable **VECTOR extension** (see
  [Vector index extension](#vector-index-extension-ladybug-018x) above) — needs network on first
  use, then cached under `~/.lbdb/extension/`. It also requires **OpenSSL 3** on the system (Windows
  needs `libssl-3-x64.dll` / `libcrypto-3-x64.dll`) — see the
  [OpenSSL 3 requirement](#openssl-3-requirement) section.
