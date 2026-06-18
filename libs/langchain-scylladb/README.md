# LangChain ScyllaDB integration

This package contains the LangChain integration with [ScyllaDB Cloud](https://cloud.scylladb.com).

- `langchain-scylladb` ([PyPI](https://pypi.org/project/langchain-scylladb/))

## Installation
```bash
pip install langchain-scylladb
```

## Usage

```python
import os
from langchain_core.documents import Document
from langchain_scylladb import ScyllaDBVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = ScyllaDBVectorStore(
    embedding=OpenAIEmbeddings(),
    host=os.environ["SCYLLA_HOST"],
    username=os.environ["SCYLLA_USER"],
    password=os.environ["SCYLLA_PASSWORD"],
    local_dc=os.environ["SCYLLA_DC"],   # e.g. "AWS_US_EAST_1"
    keyspace="my_app",
    table_name="documents",
    similarity_function="COSINE",       # or "DOT_PRODUCT" / "EUCLIDEAN"
)

# Add documents
docs = [
    Document(page_content="ScyllaDB is a high-performance NoSQL database", metadata={"source": "docs"}),
    Document(page_content="LangChain simplifies LLM apps", metadata={"source": "blog"}),
]
vectorstore.add_documents(documents=docs, ids=["1", "2"])

# Delete by IDs
vectorstore.delete(ids=["2"])

# Similarity search
results = vectorstore.similarity_search("fast database", k=2)

# Similarity search with scores
results = vectorstore.similarity_search_with_score("fast database", k=2)
for doc, score in results:
    print(f"[{score:.3f}] {doc.page_content}")

# Simple equality filter
results = vectorstore.similarity_search(
    "database", k=2, filter={"source": "docs"}
)

# Compound filter
results = vectorstore.similarity_search(
    "database",
    k=2,
    filter={"$or": [{"source": "docs"}, {"source": "blog"}]},
)

# Maximal Marginal Relevance (MMR) — balances relevance with diversity
results = vectorstore.max_marginal_relevance_search(
    "database", k=2, fetch_k=20, lambda_mult=0.5
)

# Use as a retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})
results = retriever.invoke("fast database")
```

## Contributing
See the [Contributing Guide](../../CONTRIBUTING.md).
