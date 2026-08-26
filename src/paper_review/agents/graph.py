"""LangGraph orchestration: scheduler -> research -> writer -> reviewer -> verifier,
with two independent revision loops -- one for coherence (writer <-> reviewer),
one for grounding (writer <-> verifier). Separate because passing one says
nothing about the other: a draft can be clearly written and still ungrounded,
or grounded but incoherent.

Multi-paper: a run takes a topic plus a *set* of source papers and produces
one synthesized manuscript drawing across all of them, not a review of a
single paper. research_node tags each note with which papers it actually
drew evidence from, and writer_node is prompted to cite those arXiv IDs
inline -- that citation trail is what "authenticity" means for a synthesis
across multiple sources: not just that a claim is grounded *somewhere*, but
that a reader can see which source backs which sentence.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from paper_review.config import get_llm
from paper_review.retrieval.pipeline import retrieve
from paper_review.verification.pipeline import verify_draft, all_claims_supported


class ReviewState(TypedDict):
    paper_ids: list[str]
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


def initial_state(
    paper_ids: list[str], topic: str, max_revisions: int = 2, max_verification_revisions: int = 2
) -> ReviewState:
    """The single source of truth for ReviewState's starting shape -- every
    caller (evaluation harness, API, ad-hoc scripts) needs the exact same
    zeroed fields, and hand-rolling that dict per call site is how one of
    them silently drifts out of sync with the TypedDict."""
    return {
        "paper_ids": paper_ids,
        "topic": topic,
        "subtopics": [],
        "research_notes": [],
        "draft": "",
        "review_feedback": "",
        "review_verdict": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
        "verification_feedback": "",
        "verification_passed": False,
        "verification_revision_count": 0,
        "max_verification_revisions": max_verification_revisions,
    }


class Subtopics(BaseModel):
    subtopics: list[str] = Field(description="2-4 focused research angles on the topic")


def scheduler_node(state: ReviewState) -> dict:
    llm = get_llm("cheap", schema=Subtopics)
    result = llm.invoke(f"Break this research topic into 2-4 focused angles: {state['topic']}")
    return {"subtopics": result.subtopics}


def research_node(state: ReviewState) -> dict:
    """Retrieve across the full set of source papers per subtopic -- ranked
    by relevance, not split evenly per paper -- and tag each note with which
    papers its evidence actually came from, straight from chunk metadata.
    That tag is what lets writer_node cite a specific source per claim
    instead of a vague "the literature says"."""
    llm = get_llm("cheap")
    notes = []
    for subtopic in state["subtopics"]:
        chunks = retrieve(subtopic, k=5, paper_ids=state["paper_ids"])
        if not chunks:
            continue
        context = "\n\n".join(c["text"] for c in chunks)
        sources = sorted({c["paper_id"] for c in chunks})
        response = llm.invoke(
            f"Summarize what this text says about '{subtopic}' in 2-3 sentences, "
            f"citing only what's actually present:\n\n{context}"
        )
        notes.append(f"[{subtopic}] (sources: {', '.join(sources)}) {response.content}")
    return {"research_notes": notes}


def writer_node(state: ReviewState) -> dict:
    """Draft, or revise if either reviewer or verifier left feedback.

    Instructed to cite the arXiv ID after each claim it draws from the
    notes -- the notes carry a "(sources: ...)" tag precisely so the writer
    has something concrete to cite instead of inventing a citation style.
    This doesn't make citations verifier-checked against the *specific*
    cited paper (verify_claim checks against the pooled evidence, not a
    single source -- see its docstring); it's what makes the manuscript
    read as attributable at all, prerequisite to that being checkable.

    max_tokens=3072, not 4096: a real multi-paper run needed more than the
    old 2048 (a two-paper comparative synthesis got cut off mid-section),
    but a revision call's prompt already includes the full prior draft --
    pushing the completion budget too high risks the same "request too
    large" 413 this project hit once before (see config.get_llm), since
    prompt + max_tokens share one cap. 3072 clears the truncation seen in
    practice while leaving headroom for a revision prompt that's already
    carrying a few thousand tokens of prior draft."""
    llm = get_llm("strong", max_tokens=3072)
    notes = "\n".join(state["research_notes"])

    feedback = state.get("review_feedback") or state.get("verification_feedback")
    if feedback:
        prompt = (
            f"Revise this manuscript based on the feedback below.\n\n"
            f"Draft:\n{state['draft']}\n\nFeedback:\n{feedback}\n\n"
            f"Research notes:\n{notes}"
        )
    else:
        prompt = (
            f"Write a research manuscript synthesizing what these papers say about "
            f"'{state['topic']}'. Draw connections and contrasts across sources rather "
            f"than summarizing each paper separately. After each factual claim, cite the "
            f"source arXiv ID it came from in the form (arXiv:XXXX.XXXXX), using the "
            f"'sources:' tag on the relevant research note -- if a note lists multiple "
            f"sources, cite whichever one you're actually drawing that specific claim "
            f"from.\n\nResearch notes:\n{notes}"
        )

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
    results = verify_draft(state["draft"], state["paper_ids"])
    passed = all_claims_supported(results)

    feedback = ""
    if not passed:
        failed = [
            f'- "{r.claim}" -- {r.verdict.verdict}: {r.verdict.reasoning}'
            for r in results
            if r.verdict.verdict != "supported"
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