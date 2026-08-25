"""Embed chunks into ChromaDB and retrieve relevant ones for a query.

ChromaDB's embedding_function handles the sentence-transformers calls
internally -- we never touch raw vectors directly.
"""

import chromadb
from chromadb.utils import embedding_functions

from paper_review.config import settings
from paper_review.ingestion.pipeline import Chunk, chunk_paper

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


def retrieve(query: str, k: int = 5, paper_ids: list[str] | None = None) -> list[dict]:
    """Return the k most relevant chunks for a query, optionally scoped to a
    set of papers. Chroma's $in operator handles one paper or many
    identically, so callers never need a separate single-paper code path --
    a single-paper caller just passes a one-element list."""
    collection = get_collection()
    where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
    results = collection.query(query_texts=[query], n_results=k, where=where)

    return [
        {"chunk_id": id_, "text": doc, **meta}
        for id_, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


def ensure_ingested(paper_ids: list[str]) -> None:
    """Ingest any paper_id not already present in the collection, skipping
    ones that already have chunks stored. Shared by the API and the
    evaluation harness so "has this paper actually been chunked and
    embedded yet" has exactly one implementation -- a paper silently
    missing from retrieval produces no error, just an ungrounded review
    with no obvious cause, which is a much worse failure mode than a
    slightly slower first request."""
    collection = get_collection()
    for paper_id in paper_ids:
        existing = collection.get(where={"paper_id": paper_id}, limit=1)
        if existing["ids"]:
            continue
        store_chunks(chunk_paper(paper_id))