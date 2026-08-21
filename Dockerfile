FROM python:3.12-slim

WORKDIR /app

# build-essential as a safety net for any transitive dep without a manylinux
# wheel for this python/arch combo (sentence-transformers' chain is the main
# risk) -- not confirmed necessary since this image hasn't been build-tested
# here (no docker available in the dev sandbox this was written in), but
# cheap insurance against a build that fails with no compiler available.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY scripts/ scripts/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "paper_review.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
