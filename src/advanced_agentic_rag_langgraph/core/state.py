from typing_extensions import TypedDict, Annotated
from typing import Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator

class AdvancedRAGState(TypedDict):
    """Enhanced state with query optimization and conversational support"""

    # Input
    user_question: str  # Raw user input (immutable)
    baseline_query: str  # Post-conversational-rewrite baseline (immutable after conversational_rewrite)
    conversation_history: list[dict[str, str]]  # Past conversation turns

    # Query optimization
    query_expansions: Optional[list[str]]  # Multiple query variants (regenerated per iteration, not accumulated)
    active_query: Optional[str]  # Current working query for retrieval (mutable through optimization/rewriting)

    # Retrieval strategy
    retrieval_strategy: Optional[Literal["semantic", "keyword", "hybrid"]]
    strategy_changed: Optional[bool]  # Flag to skip decide_strategy node on retry (routing signal)
    strategy_switch_reason: Optional[str]  # Explanation of why strategy was switched
    corpus_stats: Optional[dict[str, any]]  # Corpus-level statistics for strategy selection

    # Processing
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: Annotated[list[str], operator.add]
    unique_docs_list: Optional[list]  # Document objects for metadata analysis
    retrieval_quality_score: Optional[float]  # How good are the retrieved docs?

    # Retrieval Evaluation (Golden Dataset Support)
    ground_truth_doc_ids: Optional[set]  # Relevant document IDs from golden dataset
    relevance_grades: Optional[dict[str, int]]  # Optional graded relevance (doc_id -> 0-3 scale)
    retrieval_metrics: Optional[dict[str, float]]  # Recall@K, Precision@K, F1@K, nDCG@K

    # Decision making
    needs_retrieval: Optional[bool]
    is_answer_sufficient: Optional[bool]
    retrieval_attempts: Optional[int]

    # Groundedness & Hallucination Detection (RAG Triad framework)
    groundedness_score: Optional[float]  # Percentage of claims supported by context (0.0-1.0)
    has_hallucination: Optional[bool]  # Whether unsupported claims detected
    unsupported_claims: Optional[list[str]]  # List of claims not supported by context
    retry_needed: Optional[bool]  # Whether severe hallucination requires regeneration
    groundedness_severity: Optional[Literal["NONE", "MODERATE", "SEVERE"]]
    groundedness_retry_count: Optional[int]  # Number of regeneration attempts due to hallucination
    retrieval_caused_hallucination: Optional[bool]  # Flag when poor retrieval causes hallucination (triggers re-retrieval)

    # Metadata-driven adaptive retrieval
    doc_metadata_analysis: Optional[dict[str, any]]  # Analysis of retrieved document characteristics
    strategy_mismatch_rate: Optional[float]  # Percentage of docs preferring different strategy
    avg_doc_confidence: Optional[float]  # Average strategy confidence from retrieved docs
    domain_alignment_score: Optional[float]  # How well docs match query domain
    refinement_history: Optional[Annotated[list[dict[str, any]], operator.add]]  # Log of all refinements

    # Retrieval Quality Issues (replaces context sufficiency - streamlined to single source)
    retrieval_quality_reasoning: Optional[str]  # LLM explanation of retrieval quality score
    retrieval_quality_issues: Optional[list[str]]  # Specific problems: partial_coverage, missing_key_info, etc.

    # Answer Quality Evaluation (mirrors retrieval quality pattern)
    answer_quality_reasoning: Optional[str]  # LLM explanation of answer evaluation
    answer_quality_issues: Optional[list[str]]  # Specific problems: incomplete_synthesis, lacks_specificity, etc.

    # Output
    final_answer: Optional[str]
    confidence_score: Optional[float]
