"""Grounding verification: extract factual claims from a draft, independently
re-retrieve evidence for each one, and judge support with a three-way verdict.

Verification never trusts writer-reported citations -- it re-retrieves
evidence per claim itself, so there's no "coverage" metric to game by
over-citing.

Multi-paper: a claim is checked against the combined evidence pool across
every source paper, not one at a time per paper. This is a deliberate scope
cut, not an oversight -- confirming that a claim's *content* is grounded
somewhere in the source set is the load-bearing check; confirming that a
claim is grounded specifically in whichever paper the writer happened to
cite it to would need parsing citations back out of prose and matching them
to a paper_id, which is real added complexity for a narrower guarantee.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from paper_review.config import get_llm
from paper_review.retrieval.pipeline import retrieve


class ExtractedClaims(BaseModel):
    claims: list[str] = Field(description="Discrete factual assertions made in the text, one per claim")


class ClaimVerdict(BaseModel):
    verdict: Literal["supported", "contradicted", "unsupported"]
    reasoning: str = Field(description="One sentence explaining the verdict")


@dataclass
class VerifiedClaim:
    claim: str
    verdict: ClaimVerdict
    evidence_paper_ids: list[str]  # which source papers the retrieved evidence actually came from


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


def verify_claim(claim: str, paper_ids: list[str], k: int = 5) -> VerifiedClaim:
    """Independently re-retrieve evidence for one claim and judge it against
    that evidence -- not against whatever the writer says it used.

    Evidence is pooled across every paper in paper_ids -- retrieve() ranks
    by relevance regardless of source, so a claim gets checked against
    whichever passages actually match best, not an arbitrary per-paper
    split. evidence_paper_ids records which papers those passages actually
    came from, straight from the chunk metadata already returned by
    retrieve() -- not something asked of the LLM, which has no reliable way
    to introspect which of its inputs it leaned on.

    k=5, not the original 3: a ground-truth eval run (see evaluation/) found
    genuinely true claims coming back "unsupported" because the one chunk
    that actually covered them didn't make the top 3 -- a retrieval-recall
    gap, not the judge being wrong. Widening k costs a bigger prompt per
    verify_claim call (~5 chunks x ~300 tokens), comfortably inside Groq's
    per-request cap even added to the strong tier's default max_tokens."""
    chunks = retrieve(claim, k=k, paper_ids=paper_ids)
    evidence = "\n\n".join(c["text"] for c in chunks)
    evidence_paper_ids = sorted({c["paper_id"] for c in chunks})

    llm = get_llm("strong", schema=ClaimVerdict)
    verdict = llm.invoke(
        f"Evidence from the source papers:\n{evidence}\n\n"
        f"Claim to check: {claim}\n\n"
        f"Is this claim 'supported' by the evidence, 'contradicted' by it, "
        f"or 'unsupported' (evidence doesn't confirm or deny it)? "
        f"Empty or irrelevant evidence means unsupported, not supported."
    )
    return VerifiedClaim(claim=claim, verdict=verdict, evidence_paper_ids=evidence_paper_ids)


def verify_draft(draft: str, paper_ids: list[str]) -> list[VerifiedClaim]:
    """Extract and verify every claim in a draft."""
    claims = extract_claims(draft)
    return [verify_claim(claim, paper_ids) for claim in claims]


def all_claims_supported(results: list[VerifiedClaim]) -> bool:
    """Shared gate function -- called from both verifier_node and
    route_after_verification, so what counts as 'passing' only changes in
    one place. Anything short of 'supported' fails the gate, including
    'unsupported' -- treating unverifiable as passing would defeat the
    point of grounding verification."""
    return all(r.verdict.verdict == "supported" for r in results)