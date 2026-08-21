# paper-review-agents

Multi-agent scientific review generation with grounding verification.

Given an arXiv paper and a topic, this system runs a small team of LLM
agents to produce a literature-review-style write-up, revises it for
coherence, and then independently re-verifies every factual claim against
the source paper before calling it done -- the verifier never trusts the
writer's self-reported citations, it re-retrieves evidence per claim itself.

## Architecture

1. **Ingestion** (`ingestion/`) -- fetch an arXiv PDF, extract text per page,
   section-classify each page with a cheap-tier LLM call, chunk with
   deterministic hash-based IDs.
2. **Retrieval** (`retrieval/`) -- embed chunks into ChromaDB
   (sentence-transformers), retrieve top-k scoped to a paper_id.
3. **Agents** (`agents/graph.py`, LangGraph) --
   `scheduler -> research -> writer -> reviewer -> verifier`, with two
   independent revision loops: writer<->reviewer for coherence,
   writer<->verifier for grounding. Separate because passing one says
   nothing about the other.
4. **Verification** (`verification/`) -- extracts discrete factual claims
   from a draft (including citations), independently re-retrieves evidence
   per claim, and judges supported / contradicted / unsupported.
5. **Evaluation** (`evaluation/`) -- two pieces: hand-labeled ground truth
   to score the verifier's own precision/recall, and a baseline-vs-treatment
   harness comparing the pipeline with the verification loop on vs off.
6. **Serving** (`api/`, `ui/`) -- FastAPI `/review` endpoint, Streamlit
   front-end as a thin HTTP client over it.
7. **Observability** (`observability/tracer.py`) -- per-run node timing and
   token usage, attached as a LangChain callback and saved to
   `data/traces/`.

## Setup

### Option A: local venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # then fill in GROQ_API_KEY
```

### Option B: Docker

```bash
cp .env.example .env   # then fill in GROQ_API_KEY
docker compose up --build
```

API on `:8000`, UI on `:8501`. **Not build-tested** -- no Docker available
in the environment this was written in. The Dockerfile/compose file were
validated for YAML syntax and correct file references, not an actual build;
report back if `docker compose up --build` needs adjustment.

## Running

**API + UI:**

```bash
uvicorn paper_review.api.main:app --reload
streamlit run src/paper_review/ui/app.py
```

**Evaluation:**

```bash
python scripts/run_evaluation.py ground_truth   # verifier precision/recall
python scripts/run_evaluation.py matrix          # baseline vs treatment
python scripts/run_evaluation.py retry           # after a run fails on a quota wall
python scripts/run_evaluation.py report          # regenerate the report from checkpoint
```

## Known issues

- **macOS + iCloud Desktop sync**: if this repo lives under an
  iCloud-synced folder (e.g. `~/Desktop`), iCloud periodically re-applies
  the macOS "hidden" flag to the editable install's `.pth` file, and
  Python 3.13+'s `site.py` silently skips hidden `.pth` files --
  `paper_review` becomes unimportable with no obvious cause. Symptom:
  `ModuleNotFoundError: No module named 'paper_review'` despite
  `pip show paper-review-agents` looking completely fine. Fix:
  `chflags nohidden .venv/lib/python*/site-packages/__editable__.paper_review_agents-*.pth`,
  or run with `PYTHONPATH=src` to sidestep the editable-install mechanism,
  or run via Docker (fresh Linux filesystem each build, unaffected).
- **Groq's free tier has a *daily* token cap (TPD), not just per-minute.**
  A single full pipeline run (multiple review + verification revisions)
  can use tens of thousands of tokens on its own; the evaluation matrix
  can exhaust the 200k/day budget in a handful of runs. That's why
  `scripts/run_evaluation.py` checkpoints every run to JSONL and has an
  explicit `retry` command rather than auto-retrying.

## Project status

| Phase | Status |
|---|---|
| 0. Scaffold | done |
| 1. Ingestion | done |
| 2. Retrieval | done |
| 3. Agents | done |
| 4. Verifier | done |
| 5. Evaluation | ground-truth eval has real results -- 87.5% accuracy (14/16), 100% precision on both "supported" and "contradicted" (zero false confirmations either direction). Baseline-vs-treatment matrix is wired up and checkpointed but hasn't completed a full run yet -- blocked on a Groq quota window long enough to finish it. |
| 6. Ship | API, UI, and tracing done and smoke-tested. Docker written but not build-tested. Provider failover ("cost routing") deferred pending a second LLM provider key. README, this file. |

## Project structure

```
src/paper_review/
  config.py            settings + get_llm() provider dispatch (Groq)
  ingestion/pipeline.py     arXiv fetch -> extract -> section-classify -> chunk
  retrieval/pipeline.py     Chroma embed + retrieve
  agents/graph.py           the LangGraph state machine + initial_state()
  verification/pipeline.py  claim extraction + independent re-verification
  evaluation/               ground_truth.py, metrics.py, benchmark.py, harness.py, report.py
  observability/tracer.py   per-run node timing + token usage
  api/main.py               FastAPI /review, /health, /traces/{id}
  ui/app.py                 Streamlit front-end
scripts/
  run_evaluation.py     CLI for the evaluation harness
  test_*.py              ad-hoc smoke-test scripts, not a formal test suite
```
