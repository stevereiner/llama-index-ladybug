# llama-index-ladybug

[LlamaIndex](https://www.llamaindex.ai/) graph store integration for [Ladybug](https://ladybugdb.com/) — an open source, embedded graph database with Cypher query support, ACID transactions, and a built-in HNSW vector index.

Ladybug is a fork of [Kùzu](https://kuzudb.com/), extended with additional features for graph RAG workloads.

## Installation

```bash
uv pip install llama-index-ladybug
```

## Quick Start

### Property Graph (recommended)

```python
from pathlib import Path
import real_ladybug as lb
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

### Knowledge Graph (legacy API)

```python
import real_ladybug as lb
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
- [Knowledge Graph Notebook](docs/notebooks/LadybugGraphDemo.ipynb)

## Development

```bash
# Clone and set up
git clone https://github.com/<your-org>/llama-index-ladybug
cd llama-index-ladybug
uv sync --group dev

# Run tests
pytest

# Install pre-commit hooks (strips notebook outputs on commit)
pre-commit install
```

## Requirements

- Python 3.9+
- `real-ladybug >= 0.15.2`
- `llama-index-core >= 0.13.0`
