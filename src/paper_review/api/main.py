"""FastAPI serving layer: wraps the agent graph behind a single /review
endpoint that auto-ingests any papers not already in the vector store.
Callers (the Streamlit UI, or anything else) only ever need to know a
topic and a list of arXiv IDs -- the ingest/retrieve/generate staging is
an implementation detail, not something every client has to reimplement.

Each request runs the graph synchronously and blocks until the full
write-revise-verify loop finishes (can take minutes) -- there's no job
queue here. That's a deliberate scope cut for now, not an oversight: adding
background jobs and polling is real added complexity that only pays off
once something actually needs concurrent requests.
"""

import json

from fastapi import FastAPI, HTTPException
from groq import RateLimitError
from pydantic import BaseModel, Field

from paper_review.agents.graph import build_graph, initial_state
from paper_review.observability.tracer import TRACE_DIR, GraphTracer
from paper_review.retrieval.pipeline import ensure_ingested

app = FastAPI(title="paper-review-agents")


class ReviewRequest(BaseModel):
    paper_ids: list[str] = Field(
        min_length=1, description="arXiv IDs of the source papers, e.g. ['1706.03762', '1512.03385']"
    )
    topic: str = Field(description="The research topic the manuscript should synthesize")
    max_revisions: int = Field(default=2, ge=0, le=5)
    max_verification_revisions: int = Field(default=2, ge=0, le=5)


class ReviewResponse(BaseModel):
    draft: str
    subtopics: list[str]
    research_notes: list[str]
    reviewer_revision_count: int
    verification_passed: bool
    verification_revision_count: int
    trace_id: str
    total_tokens: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest) -> ReviewResponse:
    try:
        ensure_ingested(request.paper_ids)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"could not fetch/ingest one of {request.paper_ids!r}: {e}"
        ) from e

    graph = build_graph()
    tracer = GraphTracer()
    try:
        result = graph.invoke(
            initial_state(
                request.paper_ids,
                request.topic,
                request.max_revisions,
                request.max_verification_revisions,
            ),
            config={"callbacks": [tracer]},
        )
    except RateLimitError as e:
        # Groq's free tier caps tokens per *day*, not just per minute -- this
        # isn't a transient blip, and without this handler it surfaces to the
        # client as an opaque 500 with no indication of what actually failed
        # or that retrying immediately won't help.
        raise HTTPException(
            status_code=503,
            detail=(
                "The LLM provider's daily quota is exhausted for now -- this "
                "isn't a bug, it clears on its own after a wait. "
                f"Provider message: {e}"
            ),
        ) from e
    tracer.save()

    return ReviewResponse(
        draft=result["draft"],
        subtopics=result["subtopics"],
        research_notes=result["research_notes"],
        reviewer_revision_count=result["revision_count"],
        verification_passed=result["verification_passed"],
        verification_revision_count=result["verification_revision_count"],
        trace_id=tracer.run_id,
        total_tokens=tracer.summary()["total_tokens"],
    )


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no trace recorded for {trace_id!r}")
    with open(path) as f:
        return json.load(f)
