# Multi-stage Dockerfile for Advanced Agentic RAG API
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies
# Use --no-dev to skip development dependencies
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ /app/src/

# Copy evaluation data (for marker_json_v2 chunks)
COPY evaluation/corpus_chunks/ /app/evaluation/corpus_chunks/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Run the API
CMD ["uvicorn", "advanced_agentic_rag_langgraph.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
