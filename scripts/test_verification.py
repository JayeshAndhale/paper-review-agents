from paper_review.ingestion.pipeline import chunk_paper
from paper_review.retrieval.pipeline import store_chunks
from paper_review.agents.graph import build_graph

chunks = chunk_paper("1706.03762")
store_chunks(chunks)

graph = build_graph()
result = graph.invoke({
    "paper_id": "1706.03762",
    "topic": "How does the attention mechanism work in the Transformer?",
    "subtopics": [],
    "research_notes": [],
    "draft": "",
    "review_feedback": "",
    "review_verdict": "",
    "revision_count": 0,
    "max_revisions": 2,
    "verification_feedback": "",
    "verification_passed": False,
    "verification_revision_count": 0,
    "max_verification_revisions": 2,
})

print("VERIFICATION PASSED:", result["verification_passed"])
print("REVIEW REVISIONS:", result["revision_count"])
print("VERIFICATION REVISIONS:", result["verification_revision_count"])
print("\nFINAL DRAFT:\n", result["draft"])