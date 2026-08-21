"""Per-run tracing: node-level timing plus per-LLM-call token usage,
recorded via a LangChain callback and saved as one JSON trace file.

Node-name detection verified directly against this project's installed
langgraph version (not assumed): a bare StateGraph invoke with no LLM
calls confirms node spans arrive as on_chain_start with kwargs["name"]
exactly matching the string passed to add_node, plus one unrelated
top-level "LangGraph" wrapper span with no parent -- filtering to
KNOWN_NODE_NAMES excludes that wrapper along with any internal LangChain
plumbing chains.

Attach a fresh tracer per graph.invoke() call via config={"callbacks":
[tracer]} -- it's stateful, scoped to one run, and not safe to reuse.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.callbacks.base import BaseCallbackHandler

TRACE_DIR = Path("data/traces")

KNOWN_NODE_NAMES = {"scheduler", "research", "writer", "reviewer", "verifier"}


@dataclass
class Span:
    span_id: str
    name: str
    kind: str  # "node" | "llm"
    start_ts: float
    end_ts: float | None = None
    status: str = "running"  # "running" | "ok" | "error"
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def duration_ms(self) -> float | None:
        return round((self.end_ts - self.start_ts) * 1000, 1) if self.end_ts is not None else None


class GraphTracer(BaseCallbackHandler):
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.spans: dict[str, Span] = {}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        if name not in KNOWN_NODE_NAMES:
            return  # not a graph node -- skip the top-level "LangGraph" wrapper and internal plumbing
        self.spans[str(run_id)] = Span(str(run_id), name, "node", time.monotonic())

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id, "ok")

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id, "error", str(error))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or "llm_call"
        self.spans[str(run_id)] = Span(str(run_id), name, "llm", time.monotonic())

    def on_llm_end(self, response, *, run_id, **kwargs):
        span = self.spans.get(str(run_id))
        if span is not None:
            usage = (response.llm_output or {}).get("token_usage", {}) if response.llm_output else {}
            span.prompt_tokens = usage.get("prompt_tokens", 0)
            span.completion_tokens = usage.get("completion_tokens", 0)
        self._end(run_id, "ok")

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, "error", str(error))

    def _end(self, run_id, status: str, error: str = "") -> None:
        span = self.spans.get(str(run_id))
        if span is None:
            return  # a span we never saw start (e.g. filtered out) -- don't crash the run over a trace gap
        span.end_ts = time.monotonic()
        span.status = status
        span.error = error

    def summary(self) -> dict:
        node_spans = sorted((s for s in self.spans.values() if s.kind == "node"), key=lambda s: s.start_ts)
        llm_spans = [s for s in self.spans.values() if s.kind == "llm"]
        prompt_tokens = sum(s.prompt_tokens for s in llm_spans)
        completion_tokens = sum(s.completion_tokens for s in llm_spans)
        return {
            "run_id": self.run_id,
            "node_count": len(node_spans),
            "llm_call_count": len(llm_spans),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "nodes": [
                {"name": s.name, "duration_ms": s.duration_ms, "status": s.status, "error": s.error}
                for s in node_spans
            ],
        }

    def save(self, path: Path | None = None) -> Path:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = path or (TRACE_DIR / f"{self.run_id}.json")
        with open(out_path, "w") as f:
            json.dump(
                {"run_id": self.run_id, "summary": self.summary(), "spans": [asdict(s) for s in self.spans.values()]},
                f,
                indent=2,
            )
        return out_path
