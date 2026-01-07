"""Pydantic request/response models for the RAG API."""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    question: str = Field(..., min_length=1, description="The question to ask the RAG system")
    thread_id: Optional[str] = Field(
        None, description="Thread ID for multi-turn conversations. Auto-generated if not provided."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is attention mechanism?",
                    "thread_id": None,
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    answer: str = Field(..., description="The generated answer")
    confidence_score: float = Field(..., ge=0, le=1, description="Answer confidence (0-1)")
    retrieval_quality_score: float = Field(..., ge=0, le=1, description="Retrieval quality (0-1)")
    groundedness_score: float = Field(..., ge=0, le=1, description="HHEM groundedness score (0-1)")
    retrieval_attempts: int = Field(..., ge=0, description="Number of retrieval attempts")
    generation_attempts: int = Field(..., ge=1, description="Number of generation attempts used")
    retrieval_strategy: str = Field(..., description="Strategy used: semantic, keyword, or hybrid")
    query_rewritten: bool = Field(..., description="Whether the query was rewritten")
    has_hallucination: bool = Field(False, description="Whether hallucination was detected")
    unsupported_claims: Optional[list[str]] = Field(
        None, description="Claims not supported by retrieved documents"
    )
    sources: list[str] = Field(..., description="Source document filenames")
    top_chunks: list[str] = Field(..., description="Top retrieved chunks (truncated)")
    processing_time_seconds: float = Field(..., ge=0, description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status: healthy or unhealthy")


class ReadyResponse(BaseModel):
    """Response model for readiness check endpoint."""

    status: str = Field(..., description="Readiness status: ready or not_ready")
    retriever_initialized: bool = Field(..., description="Whether the retriever is initialized")
    message: Optional[str] = Field(None, description="Additional status message")


class ConfigResponse(BaseModel):
    """Response model for configuration endpoint."""

    model_tier: str = Field(..., description="Current model tier: BUDGET, BALANCED, or PREMIUM")
    vector_store_backend: str = Field(..., description="Vector store backend: faiss or azure")
    version: str = Field(..., description="API version")
