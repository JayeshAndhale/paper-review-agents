"""Fetch an arXiv paper, extract full text, and chunk it into labeled sections.

Section boundaries come from a cheap-tier LLM call per page instead of
regex heading-matching -- this removes the old repo's ToC-flattening bug
class entirely instead of patching around it. chunk_id stays deterministic
and hash-based, never LLM-generated, so re-running ingestion on the same
paper always produces the same IDs.
"""

import hashlib
import os
from dataclasses import dataclass

import arxiv
import fitz  # PyMuPDF -- package is "pymupdf", import name is "fitz"
import requests
from langchain_core.messages import HumanMessage

from paper_review.config import get_llm


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    page: int
    text: str


def fetch_paper(arxiv_id: str, download_dir: str = "./data/papers") -> tuple[str, str]:
    """Download a paper's PDF from arXiv. Returns (pdf_path, title).

    arxiv>=4.0 dropped Result.download_pdf() -- pdf_url is still a plain
    attribute, so we fetch it ourselves instead of relying on a convenience
    method that may keep changing across versions.
    """
    client = arxiv.Client()
    result = next(client.results(arxiv.Search(id_list=[arxiv_id])))

    os.makedirs(download_dir, exist_ok=True)
    pdf_path = os.path.join(download_dir, f"{arxiv_id}.pdf")

    response = requests.get(result.pdf_url)
    response.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    return pdf_path, result.title


def extract_pages(pdf_path: str) -> list[str]:
    """Return raw text for each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


SECTION_PROMPT = """This is the start of a page from an academic paper. Name the
section it belongs to (e.g. Introduction, Related Work, Methods, Results,
Discussion, Conclusion, References). If the page continues a section from the
previous page, name that section. Reply with ONLY the section name.

{excerpt}"""


def classify_section(page_text: str) -> str:
    """One cheap-tier LLM call per page to label its section."""
    llm = get_llm("cheap")
    response = llm.invoke([HumanMessage(content=SECTION_PROMPT.format(excerpt=page_text[:500]))])
    return response.content.strip()


def make_chunk_id(paper_id: str, page: int, index_on_page: int) -> str:
    """Deterministic chunk ID -- hash of position, never LLM-generated, so
    the same page/position always produces the same ID across runs."""
    raw = f"{paper_id}:{page}:{index_on_page}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def chunk_paper(arxiv_id: str, chunk_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    """Fetch, extract, section-classify, and chunk a paper end to end."""
    pdf_path, _ = fetch_paper(arxiv_id)
    pages = extract_pages(pdf_path)

    chunks: list[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        section = classify_section(page_text)
        words = page_text.split()

        index_on_page = 0
        start = 0
        while start < len(words):
            piece = " ".join(words[start : start + chunk_words])
            chunks.append(Chunk(
                chunk_id=make_chunk_id(arxiv_id, page_num, index_on_page),
                paper_id=arxiv_id,
                section=section,
                page=page_num,
                text=piece,
            ))
            start += chunk_words - overlap_words
            index_on_page += 1

    return chunks