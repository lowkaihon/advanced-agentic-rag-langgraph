"""FastAPI application for Advanced Agentic RAG."""

import os
import uuid
import io
from contextlib import asynccontextmanager, redirect_stdout

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from advanced_agentic_rag_langgraph.api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    ReadyResponse,
    ConfigResponse,
)
from advanced_agentic_rag_langgraph.orchestration import advanced_rag_graph
from advanced_agentic_rag_langgraph.core import setup_retriever
import advanced_agentic_rag_langgraph.orchestration.nodes as nodes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize retriever on startup."""
    # Startup: Initialize retriever
    print("Initializing retriever...")
    try:
        if nodes.adaptive_retriever is None:
            nodes.adaptive_retriever = setup_retriever(from_marker_json=True, verbose=False)
        print("Retriever initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize retriever on startup: {e}")
        print("Retriever will be initialized on first request")

    yield

    # Shutdown: Cleanup if needed
    print("Shutting down...")


DESCRIPTION = """
Intelligent document retrieval and question answering using LangGraph.

## Demo Corpus

This demo uses **10 landmark ML/AI research papers**:

| Paper | Area |
|-------|------|
| Attention Is All You Need | Transformer architecture |
| BERT | NLP pre-training |
| Vision Transformer (ViT) | Computer vision |
| CLIP | Vision-language |
| DALL-E 2 | Text-to-image generation |
| U-Net | Image segmentation |
| Denoising Diffusion Probabilistic Models | Generative models |
| Consistency Models | Fast diffusion |
| WGAN-GP | GAN training |
| RAPTOR | RAG techniques |

## Features

- Multi-strategy retrieval (semantic/keyword/hybrid)
- Query expansion with RRF fusion
- Two-stage reranking (CrossEncoder + LLM)
- HHEM hallucination detection
- Self-correction loops with quality gates
"""

app = FastAPI(
    title="Advanced Agentic RAG API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration (if needed later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Liveness probe - checks if the service is running.

    Used by Azure Container Apps for health monitoring.
    """
    return HealthResponse(status="healthy")


@app.get("/v1/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness probe - checks if the service is ready to accept requests.

    Verifies that the retriever is initialized and ready.
    """
    is_ready = nodes.adaptive_retriever is not None

    if is_ready:
        return ReadyResponse(
            status="ready",
            retriever_initialized=True,
            message="Service is ready to accept requests",
        )
    else:
        return ReadyResponse(
            status="not_ready",
            retriever_initialized=False,
            message="Retriever not yet initialized",
        )


@app.get("/v1/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """
    Get current configuration.

    Returns the active model tier and vector store backend.
    """
    model_tier = os.getenv("MODEL_TIER", "budget").upper()
    vector_store = os.getenv("VECTOR_STORE_BACKEND", "faiss")

    return ConfigResponse(
        model_tier=model_tier,
        vector_store_backend=vector_store,
        version="1.0.0",
    )


@app.post("/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.

    Runs the full agentic RAG pipeline including:
    - Conversational query rewriting
    - Strategy selection (semantic/keyword/hybrid)
    - Query expansion with RRF fusion
    - Two-stage reranking
    - Quality-gated answer generation
    - Hallucination detection (HHEM)
    """
    # Ensure retriever is initialized
    if nodes.adaptive_retriever is None:
        try:
            nodes.adaptive_retriever = setup_retriever(from_marker_json=True, verbose=False)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize retriever: {str(e)}",
            )

    # Generate thread ID if not provided
    thread_id = request.thread_id or str(uuid.uuid4())

    # Initial state
    initial_state = {
        "user_question": request.question,
        "baseline_query": request.question,
        "query_expansions": [],
        "active_query": request.question,
        "retrieval_strategy": "hybrid",
        "messages": [],
        "retrieved_docs": [],
        "retrieval_quality_score": 0.0,
        "is_answer_sufficient": False,
        "retrieval_attempts": 0,
        "final_answer": "",
        "confidence_score": 0.0,
    }

    # Config with thread
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run graph (suppress verbose output)
        with redirect_stdout(io.StringIO()):
            for step in advanced_rag_graph.stream(
                initial_state, config=config, stream_mode="updates"
            ):
                pass

        # Get final state
        final_state = advanced_rag_graph.get_state(config)
        final_values = final_state.values

        return QueryResponse(
            answer=final_values.get("final_answer", "No answer generated"),
            confidence_score=final_values.get("confidence_score", 0.0),
            retrieval_quality_score=final_values.get("retrieval_quality_score", 0.0),
            retrieval_attempts=final_values.get("retrieval_attempts", 0),
            retrieval_strategy=final_values.get("retrieval_strategy", "hybrid"),
            query_rewritten=(
                final_values.get("active_query") != final_values.get("baseline_query")
            ),
            query_variations=len(final_values.get("query_expansions", [])),
            thread_id=thread_id,
            has_hallucination=final_values.get("has_hallucination", False),
            unsupported_claims=final_values.get("unsupported_claims"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")
