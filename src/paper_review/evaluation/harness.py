"""Baseline vs treatment: the identical scheduler->research->writer->reviewer
graph, with the grounding-verification loop present (treatment) or absent
(baseline). Both conditions reuse the exact same node functions from
agents/graph.py, so they can never silently drift into testing different
writer/reviewer behavior -- the verification loop's presence is the only
variable this harness measures.

Baseline's draft is still scored with the same verify_draft/all_claims_supported
rubric treatment uses live, just applied once after the fact instead of
gating a revision loop. That's what makes the two conditions' numbers
comparable at all: same rubric, different only in whether it influenced
generation.

Checkpointed as append-only JSONL so a crash mid-matrix loses at most the
one run in flight, never the runs already recorded.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from langgraph.graph import END, StateGraph

from paper_review.agents.graph import (
    ReviewState,
    build_graph,
    reviewer_node,
    research_node,
    route_after_review,
    scheduler_node,
    writer_node,
)
from paper_review.evaluation.benchmark import BENCHMARK_TOPICS, BenchmarkTopic
from paper_review.verification.pipeline import all_claims_supported, verify_draft

Condition = Literal["baseline", "treatment"]

CHECKPOINT_PATH = Path("data/evaluation/results.jsonl")
DEFAULT_MAX_REVISIONS = 2


def build_baseline_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("scheduler")
    graph.add_edge("scheduler", "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "reviewer")

    # route_after_review normally routes an approved draft to "verifier",
    # which doesn't exist in this graph -- remap that one target to END,
    # everything else (including the revision loop back to "writer") as-is.
    def route_baseline(state: ReviewState) -> str:
        target = route_after_review(state)
        return END if target == "verifier" else target

    graph.add_conditional_edges("reviewer", route_baseline, {"writer": "writer", END: END})
    return graph.compile()


def score_draft(draft: str, paper_id: str) -> dict:
    results = verify_draft(draft, paper_id)
    total = len(results)
    counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for _, verdict in results:
        counts[verdict.verdict] += 1
    return {
        "total_claims": total,
        **counts,
        "grounded_rate": round(counts["supported"] / total, 4) if total else 0.0,
        "fully_grounded": all_claims_supported(results),
    }


def _initial_state(topic: BenchmarkTopic) -> dict:
    return {
        "paper_id": topic.arxiv_id,
        "topic": topic.topic_prompt,
        "subtopics": [],
        "research_notes": [],
        "draft": "",
        "review_feedback": "",
        "review_verdict": "",
        "revision_count": 0,
        "max_revisions": DEFAULT_MAX_REVISIONS,
        "verification_feedback": "",
        "verification_passed": False,
        "verification_revision_count": 0,
        "max_verification_revisions": DEFAULT_MAX_REVISIONS,
    }


@dataclass
class RunResult:
    topic_name: str
    condition: Condition
    run_number: int
    status: Literal["completed", "failed"]
    scoring: dict = field(default_factory=dict)
    reviewer_revision_count: int = 0
    verifier_revision_count: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


class ResultStore:
    """Append-only JSONL. One line per run, so a crash mid-matrix loses at
    most the one in-flight run rather than corrupting everything already
    recorded via a read-modify-write of one big file."""

    def __init__(self, path: Path = CHECKPOINT_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed_keys(self) -> set[tuple[str, str, int]]:
        if not self.path.exists():
            return set()
        with open(self.path) as f:
            return {
                (r["topic_name"], r["condition"], r["run_number"])
                for r in (json.loads(line) for line in f)
            }

    def append(self, result: RunResult) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def clear_failed(self) -> int:
        """Drop every 'failed' record so those (topic, condition, run)
        combinations become eligible for a fresh attempt on the next
        run_matrix() call. Explicit and separate from run_matrix itself --
        a failure isn't retried automatically, since re-running right after
        a TPD wall just re-hits the same wall and burns the little quota
        that has trickled back finding that out again."""
        records = self.all_results()
        kept = [r for r in records if r["status"] != "failed"]
        dropped = len(records) - len(kept)
        with open(self.path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        return dropped

    def all_results(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f]


def run_one(topic: BenchmarkTopic, condition: Condition, run_number: int) -> RunResult:
    graph = build_baseline_graph() if condition == "baseline" else build_graph()
    start = time.monotonic()
    try:
        result = graph.invoke(_initial_state(topic))
        scoring = score_draft(result["draft"], topic.arxiv_id)
    except Exception as e:
        return RunResult(topic.name, condition, run_number, status="failed", error=repr(e))

    return RunResult(
        topic_name=topic.name,
        condition=condition,
        run_number=run_number,
        status="completed",
        scoring=scoring,
        reviewer_revision_count=result.get("revision_count", 0),
        verifier_revision_count=result.get("verification_revision_count", 0),
        elapsed_seconds=round(time.monotonic() - start, 1),
    )


def run_matrix(
    topics: list[BenchmarkTopic] = BENCHMARK_TOPICS,
    runs_per_condition: int = 1,
    store: ResultStore | None = None,
) -> list[RunResult]:
    """Resumable: skips any (topic, condition, run_number) already
    checkpointed, including failed runs -- a failure isn't retried
    automatically since re-running immediately after a rate-limit or quota
    error just re-hits the same wall."""
    store = store or ResultStore()
    done = store.completed_keys()
    results = []

    for topic in topics:
        for condition in ("baseline", "treatment"):
            for run_number in range(1, runs_per_condition + 1):
                key = (topic.name, condition, run_number)
                if key in done:
                    print(f"skip (already recorded): {key}")
                    continue

                print(f"running {topic.name} / {condition} / run {run_number} ...")
                result = run_one(topic, condition, run_number)
                store.append(result)
                results.append(result)

                if result.status == "failed":
                    print(f"  FAILED after {result.elapsed_seconds}s: {result.error}")
                else:
                    print(f"  done in {result.elapsed_seconds}s: {result.scoring}")

    return results
