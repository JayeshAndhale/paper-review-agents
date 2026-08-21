"""Sanity check: GROQ_API_KEY is live and get_llm() works end to end."""

from paper_review.config import get_llm

llm = get_llm("cheap")
response = llm.invoke("What is the capital of France?")
print(response.content)