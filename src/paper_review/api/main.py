"""FastAPI serving layer: wraps the agent graph behind a single /review
endpoint that auto-ingests a paper on first request. Callers (the Streamlit
UI, or anything else) only ever need to know an arXiv ID and a topic -- the
ingest/retrieve/generate staging is an implementation detail, not something
every client has to reimplement.

Each request runs the graph synchronously and blocks until the full
review-revise-verify loop finishes (can take minutes) -- there's no job
queue here. That's a deliberate scope cut for now, not an oversight: adding
background jobs and polling is real added complexity that only pays off
once something actually needs concurrent requests.
"""

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from paper_review.agents.graph import build_graph, initial_state
from paper_review.ingestion.pipeline import chunk_paper
from paper_review.observability.tracer import TRACE_DIR, GraphTracer
from paper_review.retrieval.pipeline import get_collection, store_chunks

app = FastAPI(title="paper-review-agents")


class ReviewRequest(BaseModel):
    paper_id: str = Field(description="arXiv ID, e.g. 1706.03762")
    topic: str = Field(description="The question or angle the review should address")
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


def _ensure_ingested(paper_id: str) -> None:
    """Skip re-ingestion if this paper_id already has chunks stored --
    chunk_paper does a page-by-page LLM call per page, so re-running it on
    every request would be a lot of wasted latency and quota for a paper
    that's already in the vector store."""
    collection = get_collection()
    existing = collection.get(where={"paper_id": paper_id}, limit=1)
    if existing["ids"]:
        return
    chunks = chunk_paper(paper_id)
    store_chunks(chunks)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest) -> ReviewResponse:
    try:
        _ensure_ingested(request.paper_id)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"could not fetch/ingest paper {request.paper_id!r}: {e}"
        ) from e

    graph = build_graph()
    tracer = GraphTracer()
    result = graph.invoke(
        initial_state(
            request.paper_id,
            request.topic,
            request.max_revisions,
            request.max_verification_revisions,
        ),
        config={"callbacks": [tracer]},
    )
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
