"""Embed chunks into ChromaDB and retrieve relevant ones for a query.

ChromaDB's embedding_function handles the sentence-transformers calls
internally -- we never touch raw vectors directly.
"""

import chromadb
from chromadb.utils import embedding_functions

from paper_review.config import settings
from paper_review.ingestion.pipeline import Chunk

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.embedding_model
)


def get_collection(name: str = "paper_chunks"):
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(name=name, embedding_function=_embedding_fn)


def store_chunks(chunks: list[Chunk]) -> None:
    """Embed and upsert chunks. Upsert, not add -- re-running ingestion on
    the same paper overwrites cleanly instead of duplicating, which only
    works because chunk_id is deterministic, not random."""
    collection = get_collection()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"paper_id": c.paper_id, "section": c.section, "page": c.page} for c in chunks],
    )


def retrieve(query: str, k: int = 5, paper_id: str | None = None) -> list[dict]:
    """Return the k most relevant chunks for a query, optionally scoped to one paper."""
    collection = get_collection()
    where = {"paper_id": paper_id} if paper_id else None
    results = collection.query(query_texts=[query], n_results=k, where=where)

    return [
        {"chunk_id": id_, "text": doc, **meta}
        for id_, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]