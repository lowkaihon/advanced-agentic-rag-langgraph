from typing_extensions import TypedDict, Annotated
from typing import Literal, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator

class AdvancedRAGState(TypedDict):
    """
    State schema for advanced agentic RAG workflow with distributed decision-making.

    State Management Patterns:
    - add_messages: Message history (idempotent, deduplicates by ID)
    - operator.add: Documents and refinement history (accumulate across iterations)
    - Direct replacement: Query expansions (regenerated per iteration for expansion-query alignment)
    """

    # === INPUT & INITIALIZATION ===
    user_question: str  # Raw user input
    baseline_query: str  # Conversationally-rewritten query (set once by conversational_rewrite node)
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history with automatic deduplication

    # === QUERY LIFECYCLE ===
    active_query: Optional[str]  # Current working query (semantic, human-readable, evolves through rewrites)
    retrieval_query: Optional[str]  # Algorithm-optimized query for retrieval (set by query_expansion_node for ALL paths)
    query_expansions: Optional[list[str]]  # Query variants for multi-query fusion (generated from optimized retrieval_query)

    # === STRATEGY SELECTION & ADAPTATION ===
    retrieval_strategy: Optional[Literal["semantic", "keyword", "hybrid"]]  # Selected approach based on query and corpus analysis
    corpus_stats: Optional[dict[str, Any]]  # Document profiling: technical_density, domain_distribution, has_code/math
    strategy_changed: Optional[bool]  # Signals early switch happened (for revert validation)
    strategy_switch_reason: Optional[str]  # Content-driven explanation (e.g., "off_topic detected -> keyword")
    previous_strategy: Optional[Literal["semantic", "keyword", "hybrid"]]  # Strategy before early switch (for revert)
    previous_quality_score: Optional[float]  # Quality score before early switch (for revert comparison)

    # === RETRIEVAL EXECUTION ===
    retrieved_docs: Annotated[list[str], operator.add]  # Accumulated document content across retrieval attempts
    unique_docs_list: Optional[list]  # Deduplicated Document objects with metadata (for reranking/analysis)
    retrieval_attempts: Optional[int]  # Retrieval attempt counter: tracks attempts across rewrites/strategy switches (max 2, resets per user question)

    # === QUALITY ASSESSMENT ===
    # Retrieval Quality (LLM-as-judge evaluation)
    retrieval_quality_score: Optional[float]  # 0.0-1.0 score from structured LLM evaluation
    retrieval_quality_reasoning: Optional[str]  # LLM explanation of quality score
    retrieval_quality_issues: Optional[list[str]]  # Issues: partial_coverage, missing_key_info, off_topic, wrong_domain, etc.
    keywords_to_inject: Optional[list[str]]  # Non-tautological keywords to inject into query (if quality < 0.6)

    # Answer Quality (vRAG-Eval framework with adaptive thresholds)
    answer_quality_reasoning: Optional[str]  # LLM explanation from answer evaluation
    answer_quality_issues: Optional[list[str]]  # Issues: incomplete_synthesis, lacks_specificity, missing_details, etc.
    is_answer_sufficient: Optional[bool]  # Quality gate: proceed to output or retry
    is_refusal: Optional[bool]  # Whether LLM refused to answer due to insufficient context (terminal state)

    # === GROUNDEDNESS & HALLUCINATION (HHEM-based detection) ===
    groundedness_score: Optional[float]  # Percentage of claims supported by context (0.0-1.0)
    has_hallucination: Optional[bool]  # Whether unsupported claims detected via HHEM
    unsupported_claims: Optional[list[str]]  # Specific claims failing HHEM verification (for targeted regeneration)

    # === GENERATION RETRY (Unified retry handling) ===
    generation_attempts: Optional[int]  # Generation attempt counter (max 3 total attempts = initial + 2 retries, resets per user question)
    retry_feedback: Optional[str]  # Combined groundedness + quality feedback for regeneration
    previous_answer: Optional[str]  # Previous answer for retry context (enables targeted correction)

    # === EVALUATION METRICS (Golden Dataset Support) ===
    ground_truth_doc_ids: Optional[list]  # Relevant document IDs from test set
    relevance_grades: Optional[dict[str, int]]  # Graded relevance per doc (0-3 scale for nDCG)
    retrieval_metrics: Optional[dict[str, float]]  # Recall@K, Precision@K, F1@K, nDCG, MRR, Hit Rate

    # === OUTPUT ===
    final_answer: Optional[str]
    confidence_score: Optional[float]  # Answer confidence from quality evaluation (0.0-1.0)
