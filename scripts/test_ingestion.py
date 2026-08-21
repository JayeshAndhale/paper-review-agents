from paper_review.ingestion.pipeline import chunk_paper

chunks = chunk_paper("1706.03762")  # "Attention Is All You Need" -- small, fast test
print(f"{len(chunks)} chunks")
print(chunks[0])