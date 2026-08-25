"""Grounding verification: extract factual claims from a draft, independently
re-retrieve evidence for each one, and judge support with a three-way verdict.

Verification never trusts writer-reported citations -- it re-retrieves
evidence per claim itself, so there's no "coverage" metric to game by
over-citing.
"""

from typing import Literal

from pydantic import BaseModel, Field

from paper_review.config import get_llm
from paper_review.retrieval.pipeline import retrieve


class ExtractedClaims(BaseModel):
    claims: list[str] = Field(description="Discrete factual assertions made in the text, one per claim")


class ClaimVerdict(BaseModel):
    verdict: Literal["supported", "contradicted", "unsupported"]
    reasoning: str = Field(description="One sentence explaining the verdict")


def extract_claims(draft: str) -> list[str]:
    """One LLM call to pull out discrete factual claims from the draft,
    including citations. A fabricated reference is exactly as checkable --
    and exactly as important to catch -- as a fabricated fact in prose, so
    it must not be filtered out as 'structural commentary.'

    Explicitly skips the review's own scope statements ("this review does
    not cover X", "Y falls outside this analysis") -- those describe the
    review's structure, not a fact about the source paper, so they can
    never be supported or contradicted by retrieved evidence and would
    only ever land as a false 'unsupported' against a verifier that's
    actually working correctly."""
    llm = get_llm("cheap", schema=ExtractedClaims, max_tokens=2048)
    result = llm.invoke(
        f"List the discrete factual claims made in this text -- each one a single "
        f"assertion that could be checked against a source. Skip opinions and "
        f"transitions. Do NOT skip citations or references: for each one listed, "
        f"extract a claim of the form 'a source titled X exists and discusses Y', "
        f"so fabricated citations get checked exactly like any other factual claim. "
        f"Also skip statements about the review's own scope or structure (e.g. "
        f"'this review does not cover X', 'Y is outside the scope of this analysis') "
        f"-- those describe the write-up itself, not the source paper, and can never "
        f"be checked against it.\n\n{draft}"
    )
    return result.claims


def verify_claim(claim: str, paper_id: str, k: int = 5) -> ClaimVerdict:
    """Independently re-retrieve evidence for one claim and judge it against
    that evidence -- not against whatever the writer says it used.

    k=5, not the original 3: a ground-truth eval run (see evaluation/) found
    genuinely true claims coming back "unsupported" because the one chunk
    that actually covered them didn't make the top 3 -- a retrieval-recall
    gap, not the judge being wrong. Widening k costs a bigger prompt per
    verify_claim call (~5 chunks x ~300 tokens), comfortably inside Groq's
    per-request cap even added to the strong tier's default max_tokens."""
    chunks = retrieve(claim, k=k, paper_id=paper_id)
    evidence = "\n\n".join(c["text"] for c in chunks)

    llm = get_llm("strong", schema=ClaimVerdict)
    return llm.invoke(
        f"Evidence from the source paper:\n{evidence}\n\n"
        f"Claim to check: {claim}\n\n"
        f"Is this claim 'supported' by the evidence, 'contradicted' by it, "
        f"or 'unsupported' (evidence doesn't confirm or deny it)? "
        f"Empty or irrelevant evidence means unsupported, not supported."
    )


def verify_draft(draft: str, paper_id: str) -> list[tuple[str, ClaimVerdict]]:
    """Extract and verify every claim in a draft. Returns (claim, verdict) pairs."""
    claims = extract_claims(draft)
    return [(claim, verify_claim(claim, paper_id)) for claim in claims]


def all_claims_supported(results: list[tuple[str, ClaimVerdict]]) -> bool:
    """Shared gate function -- called from both verifier_node and
    route_after_verification, so what counts as 'passing' only changes in
    one place. Anything short of 'supported' fails the gate, including
    'unsupported' -- treating unverifiable as passing would defeat the
    point of grounding verification."""
    return all(verdict.verdict == "supported" for _, verdict in results)