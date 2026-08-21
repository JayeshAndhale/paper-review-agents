"""LangGraph orchestration: scheduler -> research -> writer -> reviewer -> verifier,
with two independent revision loops -- one for coherence (writer <-> reviewer),
one for grounding (writer <-> verifier). Separate because passing one says
nothing about the other: a draft can be clearly written and still ungrounded,
or grounded but incoherent.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from paper_review.config import get_llm
from paper_review.retrieval.pipeline import retrieve
from paper_review.verification.pipeline import verify_draft, all_claims_supported


class ReviewState(TypedDict):
    paper_id: str
    topic: str
    subtopics: list[str]
    research_notes: list[str]
    draft: str
    review_feedback: str
    review_verdict: str
    revision_count: int
    max_revisions: int
    verification_feedback: str
    verification_passed: bool
    verification_revision_count: int
    max_verification_revisions: int


class Subtopics(BaseModel):
    subtopics: list[str] = Field(description="2-4 focused research angles on the topic")


def scheduler_node(state: ReviewState) -> dict:
    llm = get_llm("cheap", schema=Subtopics)
    result = llm.invoke(f"Break this review topic into 2-4 focused research angles: {state['topic']}")
    return {"subtopics": result.subtopics}


def research_node(state: ReviewState) -> dict:
    llm = get_llm("cheap")
    notes = []
    for subtopic in state["subtopics"]:
        chunks = retrieve(subtopic, k=3, paper_id=state["paper_id"])
        context = "\n\n".join(c["text"] for c in chunks)
        response = llm.invoke(
            f"Summarize what this text says about '{subtopic}' in 2-3 sentences, "
            f"citing only what's actually present:\n\n{context}"
        )
        notes.append(f"[{subtopic}] {response.content}")
    return {"research_notes": notes}


def writer_node(state: ReviewState) -> dict:
    """Draft, or revise if either reviewer or verifier left feedback."""
    llm = get_llm("strong", max_tokens=2048)
    notes = "\n".join(state["research_notes"])

    feedback = state.get("review_feedback") or state.get("verification_feedback")
    if feedback:
        prompt = (
            f"Revise this draft based on the feedback below.\n\n"
            f"Draft:\n{state['draft']}\n\nFeedback:\n{feedback}\n\n"
            f"Research notes:\n{notes}"
        )
    else:
        prompt = f"Write a review section on '{state['topic']}' based on these research notes:\n\n{notes}"

    response = llm.invoke(prompt)
    # clear both -- a fresh draft shouldn't carry stale feedback into the next check
    return {"draft": response.content, "review_feedback": "", "verification_feedback": ""}


class ReviewVerdict(BaseModel):
    verdict: str = Field(description="'approved' or 'needs_revision'")
    feedback: str = Field(description="Specific, actionable feedback if needs_revision, else empty")


def reviewer_node(state: ReviewState) -> dict:
    """Coherence and structure only -- NOT factual grounding, that's verifier_node."""
    llm = get_llm("strong", schema=ReviewVerdict)
    result = llm.invoke(
        f"Review this draft for clarity, coherence, and structure only "
        f"(not factual accuracy):\n\n{state['draft']}"
    )
    return {
        "review_feedback": result.feedback,
        "review_verdict": result.verdict,
        "revision_count": state["revision_count"] + 1,
    }


def route_after_review(state: ReviewState) -> str:
    if state["review_verdict"] == "needs_revision" and state["revision_count"] < state["max_revisions"]:
        return "writer"
    return "verifier"


def verifier_node(state: ReviewState) -> dict:
    """Grounding check: extract claims, re-retrieve evidence per claim
    independently, verify. Doesn't trust writer-reported citations."""
    results = verify_draft(state["draft"], state["paper_id"])
    passed = all_claims_supported(results)

    feedback = ""
    if not passed:
        failed = [
            f'- "{claim}" -- {verdict.verdict}: {verdict.reasoning}'
            for claim, verdict in results
            if verdict.verdict != "supported"
        ]
        feedback = "These claims aren't grounded in the source. Fix or remove them:\n" + "\n".join(failed)

    return {
        "verification_passed": passed,
        "verification_feedback": feedback,
        "verification_revision_count": state["verification_revision_count"] + 1,
    }


def route_after_verification(state: ReviewState) -> str:
    if not state["verification_passed"] and state["verification_revision_count"] < state["max_verification_revisions"]:
        return "writer"
    return END


def build_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("scheduler")
    graph.add_edge("scheduler", "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review, {"writer": "writer", "verifier": "verifier"})
    graph.add_conditional_edges("verifier", route_after_verification, {"writer": "writer", END: END})

    return graph.compile()