from paper_review.ingestion.pipeline import chunk_paper
from paper_review.retrieval.pipeline import store_chunks, retrieve

chunks = chunk_paper("1706.03762")
store_chunks(chunks)

results = retrieve("what is the attention mechanism", k=3)
for r in results:
    print(r["section"], "| page", r["page"], "|", r["text"][:100])