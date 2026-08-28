# Milestone 1: minimal foundation image.
# No GPU/CUDA support yet — that lands when a GPU-dependent component
# (embeddings, reranking, generation) is actually implemented.

FROM python:3.11-slim

WORKDIR /workspace

# System deps kept to the bare minimum needed to build Python packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e .

# Sanity check: fail the build if the package cannot be imported.
RUN python -c "import evidencerag; print(evidencerag.__version__)"

CMD ["python", "-c", "import evidencerag; print(f'EvidenceRAG {evidencerag.__version__} foundation image OK')"]
