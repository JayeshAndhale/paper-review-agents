"""Manual smoke test for the full multi-paper pipeline: two source papers,
one synthesis topic spanning both."""

from paper_review.agents.graph import build_graph, initial_state
from paper_review.retrieval.pipeline import ensure_ingested

paper_ids = ["1706.03762", "1512.03385"]
ensure_ingested(paper_ids)

graph = build_graph()
result = graph.invoke(
    initial_state(
        paper_ids,
        "Compare how the Transformer and ResNet each solve a depth/context "
        "problem in deep learning -- one lets every layer see the whole "
        "sequence directly via attention, the other lets gradients skip "
        "layers via residual connections.",
    )
)

print("VERIFICATION PASSED:", result["verification_passed"])
print("REVIEW REVISIONS:", result["revision_count"])
print("VERIFICATION REVISIONS:", result["verification_revision_count"])
print("\nRESEARCH NOTES:\n", "\n".join(result["research_notes"]))
print("\nFINAL DRAFT:\n", result["draft"])
