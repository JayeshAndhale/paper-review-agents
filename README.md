# paper-review-agents

Multi-paper research synthesis with grounding verification.

Given a research topic and a set of arXiv papers, this system runs a small
team of LLM agents to write a manuscript synthesizing what those papers say
about the topic -- drawing connections and contrasts across sources, citing
which paper backs each claim -- then independently re-verifies every factual
claim against the source papers before calling it done. The verifier never
trusts the writer's self-reported citations; it re-retrieves evidence per
claim itself, pooled across every source paper.

## Architecture

1. **Ingestion** (`ingestion/`) -- fetch an arXiv PDF, extract text per page,
   section-classify each page with a cheap-tier LLM call, chunk with
   deterministic hash-based IDs.
2. **Retrieval** (`retrieval/`) -- embed chunks into ChromaDB
   (sentence-transformers), retrieve top-k scoped to a *set* of paper_ids
   (Chroma's `$in` filter), ranked by relevance across all of them rather
   than split evenly per paper. `ensure_ingested()` ingests whichever of a
   requested set isn't in the store yet.
3. **Agents** (`agents/graph.py`, LangGraph) --
   `scheduler -> research -> writer -> reviewer -> verifier`, with two
   independent revision loops: writer<->reviewer for coherence,
   writer<->verifier for grounding. Separate because passing one says
   nothing about the other. `research_node` tags each note with which
   papers its evidence actually came from; `writer_node` is prompted to
   cite that paper's arXiv ID inline per claim -- that citation trail is
   what makes a multi-source synthesis attributable, not just grounded
   somewhere unspecified.
4. **Verification** (`verification/`) -- extracts discrete factual claims
   from a draft (including citations), independently re-retrieves evidence
   for each one pooled across every source paper, and judges supported /
   contradicted / unsupported. (Checking a claim against the specific paper
   it was cited to, rather than the pooled set, is a real possible
   extension -- not built, since it needs parsing citations back out of
   prose.)
5. **Evaluation** (`evaluation/`) -- two pieces: hand-labeled ground truth
   to score the verifier's own precision/recall, and a baseline-vs-treatment
   harness comparing the pipeline with the verification loop on vs off.
6. **Serving** (`api/`, `ui/`) -- FastAPI `/review` endpoint (topic + list of
   arXiv IDs), Streamlit front-end as a thin HTTP client over it.
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
| 5. Evaluation | done. Ground-truth eval: 93.75% accuracy (15/16) after widening verify_claim's retrieval k and filtering meta-commentary out of claim extraction (up from 87.5%). Baseline-vs-treatment: verification loop improves grounded rate from 57.1% (baseline, no verifier) to 85.2% (treatment) on the transformer_attention topic -- proves the loop is actually earning its keep, not just adding latency. A second topic (ResNet) is checkpointed as failed pending Groq quota; retryable anytime with `run_evaluation.py retry`, not blocking. |
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
