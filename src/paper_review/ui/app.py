"""Streamlit front-end -- a thin HTTP client over the FastAPI service, not
a second copy of the graph logic. Talking to the API instead of importing
build_graph() directly means the UI exercises exactly what's actually
served; if the API's contract changes, the UI breaks loudly against a real
HTTP error instead of silently drifting onto a different code path.

Run alongside the API:
    uvicorn paper_review.api.main:app --reload
    streamlit run src/paper_review/ui/app.py
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Paper Review Agents", layout="wide")
st.title("Paper Review Agents")
st.caption("Multi-agent scientific review generation with grounding verification")

with st.form("review_form"):
    paper_id = st.text_input("arXiv ID", placeholder="1706.03762")
    topic = st.text_area(
        "Review topic",
        placeholder="How does the attention mechanism work in the Transformer?",
    )
    col1, col2 = st.columns(2)
    max_revisions = col1.number_input("Max reviewer revisions", min_value=0, max_value=5, value=2)
    max_verification_revisions = col2.number_input(
        "Max verification revisions", min_value=0, max_value=5, value=2
    )
    submitted = st.form_submit_button("Generate review")

if submitted:
    if not paper_id.strip() or not topic.strip():
        st.error("Both arXiv ID and topic are required.")
    else:
        with st.spinner("Running ingestion + multi-agent review (can take a few minutes)..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/review",
                    json={
                        "paper_id": paper_id.strip(),
                        "topic": topic.strip(),
                        "max_revisions": max_revisions,
                        "max_verification_revisions": max_verification_revisions,
                    },
                    timeout=900,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                detail = getattr(e.response, "text", str(e)) if hasattr(e, "response") and e.response else str(e)
                st.error(f"Request to the API failed: {detail}")
            else:
                st.session_state["result"] = response.json()

if "result" in st.session_state:
    result = st.session_state["result"]

    if result["verification_passed"]:
        st.success(
            f"Fully grounded after {result['verification_revision_count']} verification revision(s)"
        )
    else:
        st.warning(
            f"NOT fully grounded -- hit the revision cap "
            f"({result['verification_revision_count']} verification revision(s))"
        )
    st.caption(f"{result['reviewer_revision_count']} reviewer revision(s)")

    st.markdown(result["draft"])

    with st.expander("Subtopics + research notes"):
        st.write(result["subtopics"])
        for note in result["research_notes"]:
            st.write(note)
