from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from advanced_agentic_rag_langgraph.core import AdvancedRAGState
from advanced_agentic_rag_langgraph.orchestration.nodes import (
    conversational_rewrite_node,
    query_expansion_node,
    decide_retrieval_strategy_node,
    retrieve_with_expansion_node,
    rewrite_and_refine_node,
    answer_generation_with_quality_node,
    groundedness_check_node,
    evaluate_answer_with_retrieval_node,
)
from advanced_agentic_rag_langgraph.retrieval.query_optimization import optimize_query_for_strategy
from typing import Literal


# ========== HELPER FUNCTIONS ==========

def select_next_strategy(current: str, issues: list[str]) -> str:
    """
    Content-driven strategy selection for early switching.

    Maps retrieval quality issues to optimal strategies based on CRAG patterns.
    Called when route_after_retrieval detects fundamental strategy mismatch.

    Args:
        current: Current retrieval strategy ("semantic", "keyword", or "hybrid")
        issues: List of detected retrieval quality issues

    Returns:
        Next strategy to try based on content analysis
    """
    if "off_topic" in issues or "wrong_domain" in issues:
        # Off-topic results indicate need for precision → keyword search
        # If already using keyword, try hybrid for balance
        return "keyword" if current != "keyword" else "hybrid"

    # Default fallback (shouldn't normally reach here, but safe fallback)
    return "semantic" if current == "hybrid" else "hybrid"


# ========== QUERY OPTIMIZATION ROUTING ==========

def route_after_query_expansion(state: AdvancedRAGState) -> Literal["decide_strategy", "retrieve_with_expansion"]:
    """
    Route based on whether this is initial expansion or retry with strategy change.

    RAG-Fusion pattern: Strategy-agnostic expansions generated first, then decide optimal strategy.

    Routing logic:
    - Initial flow: No strategy yet → route to decide_strategy
    - Retry flow: Strategy already changed upstream → bypass decide_strategy, go straight to retrieve
    """
    if state.get("strategy_changed", False):
        # Strategy changed, skip decide_strategy (already set in route_after_evaluation)
        return "retrieve_with_expansion"
    else:
        # Initial flow, proceed to strategy selection
        return "decide_strategy"


# ========== RETRIEVAL ROUTING ==========

def route_after_retrieval(state: AdvancedRAGState) -> Literal["answer_generation", "rewrite_and_refine", "query_expansion"]:
    """
    Pure router: Route based on retrieval quality with content-driven early strategy switching.

    Implements dual-tier strategy switching:
    - Early tier (here): Detects obvious strategy mismatches (off_topic, wrong_domain)
    - Late tier (route_after_evaluation): Handles subtle insufficiency after answer generation

    Research-backed CRAG pattern: confidence-based action triggering.

    Note: Pure function - only reads state and returns routing decision.
    State updates for early strategy switching happen in query_expansion_node.
    """
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])

    # Good quality: proceed to answer generation
    if quality > 0.6:
        return "answer_generation"

    # Max attempts reached: forced to proceed
    if attempts >= 2:
        return "answer_generation"

    # Poor quality: Content-driven decision
    # Strategy mismatch indicators suggest fundamental approach problem, not query problem
    if "off_topic" in issues or "wrong_domain" in issues:
        # Route to query_expansion for early strategy switch
        # query_expansion_node will detect this scenario and update state accordingly
        return "query_expansion"
    else:
        # Other issues (partial_coverage, missing_key_info, etc.) may benefit from query rewriting
        return "rewrite_and_refine"


# ========== ANSWER GENERATION ROUTING ==========

def route_after_groundedness(state: AdvancedRAGState) -> Literal["answer_generation", "evaluate_answer"]:
    """
    Pure router: Route based on groundedness with retrieval quality awareness and false positive protection.

    Three-tier threshold strategy:
    - NONE (0.8-1.0): No hallucination, proceed
    - MODERATE (0.6-0.8): Likely NLI false positive, proceed without retry
    - SEVERE (<0.6): Root cause detection:
      * Good retrieval + low groundedness → LLM hallucination (retry generation)
      * Poor retrieval + low groundedness → Retrieval-caused (flag for re-retrieval)

    Research-backed approach:
    - Protects against over-conservative NLI detector (zero-shot F1: 0.65-0.70)
    - Re-retrieval reduces hallucination 46% more than regeneration when context is the issue

    Note: Pure function - only reads state and returns routing decision.
    State updates for retry counter and re-retrieval flags happen in target nodes.
    """
    retry_needed = state.get("retry_needed", False)
    retry_count = state.get("groundedness_retry_count", 0)
    groundedness_score = state.get("groundedness_score", 1.0)
    retrieval_quality = state.get("retrieval_quality_score", 0.7)

    # Check if retry limit reached
    if retry_count >= 2:
        return "evaluate_answer"

    # MODERATE severity (0.6-0.8): Likely NLI false positive
    # Empirical evidence: Correct facts marked as unsupported (110M parameters, 15% MLM)
    # Proceed without retry to avoid degrading correct answers
    if 0.6 <= groundedness_score < 0.8:
        print(f"\n{'='*60}")
        print(f"GROUNDEDNESS WARNING (Likely NLI False Positive)")
        print(f"Score: {groundedness_score:.0%} (MODERATE - not blocking)")
        print(f"Known issue: Zero-shot NLI is over-conservative")
        print(f"Action: Proceeding to evaluation without retry")
        print(f"{'='*60}\n")
        return "evaluate_answer"

    # SEVERE groundedness issue (score < 0.6): Root cause detection
    if retry_needed and retry_count < 2:
        # Distinguish: Is this an LLM problem or a retrieval problem?

        if retrieval_quality >= 0.6:
            # Good context, bad generation → Genuine LLM hallucination
            # Retry generation with stricter grounding instructions
            # answer_generation_node will increment retry counter
            print(f"\n{'='*60}")
            print(f"GROUNDEDNESS RETRY (LLM Hallucination)")
            print(f"Groundedness: {groundedness_score:.0%}")
            print(f"Retrieval quality: {retrieval_quality:.0%} (GOOD)")
            print(f"Root cause: LLM invented facts despite good context")
            print(f"Action: Regenerating with stricter grounding")
            print(f"Retry count: {retry_count + 1}/2")
            print(f"{'='*60}\n")

            return "answer_generation"
        else:
            # Poor context → Hallucination due to missing/insufficient information
            # Research: Re-retrieval superior to regeneration (46% hallucination reduction)
            # evaluate_answer_node will set re-retrieval flags
            print(f"\n{'='*60}")
            print(f"GROUNDEDNESS ISSUE (Retrieval-Caused)")
            print(f"Groundedness: {groundedness_score:.0%}")
            print(f"Retrieval quality: {retrieval_quality:.0%} (POOR)")
            print(f"Root cause: LLM filled gaps due to insufficient context")
            print(f"Action: Flagging for re-retrieval (not regeneration)")
            print(f"Research: Re-retrieval > regeneration for context gaps")
            print(f"{'='*60}\n")

            return "evaluate_answer"

    # Default: proceed to evaluation
    return "evaluate_answer"


# ========== EVALUATION ROUTING ==========

def route_after_evaluation(state: AdvancedRAGState) -> Literal["query_expansion", "END"]:
    """
    Pure router: Route based on answer evaluation with content-driven strategy switching.

    Uses retrieval quality issues (content-based) to intelligently select
    next retrieval strategy, following research-backed CRAG/Self-RAG patterns.

    Priority checks:
    1. Retrieval-caused hallucination → Force re-retrieval with strategy change
    2. Answer sufficient → END
    3. Answer insufficient → Content-driven strategy switching

    Note: Pure function - only reads state and returns routing decision.
    State updates for strategy switching happen in query_expansion_node.
    """
    # PRIORITY CHECK: Retrieval-caused hallucination detected
    # Research: Re-retrieval reduces hallucination 46% more than regeneration
    if state.get("retrieval_caused_hallucination"):
        retrieval_attempts = state.get("retrieval_attempts", 0)

        if retrieval_attempts < 3:
            # Route to query_expansion for re-retrieval with strategy change
            # query_expansion_node will detect this scenario and update state accordingly
            return "query_expansion"
        else:
            # Max attempts reached, give up
            return END

    # Standard evaluation flow
    if state.get("is_answer_sufficient"):
        return END
    elif state.get("retrieval_attempts", 0) < 3:  # Max 3 attempts
        # Route to query_expansion for content-driven strategy switching
        # query_expansion_node will detect this scenario and update state accordingly
        # (no state mutations here - router remains pure)
        return "query_expansion"
    else:
        return END


# ========== GRAPH BUILDER ==========

def build_advanced_rag_graph():
    """Build complete advanced RAG graph with all techniques"""
    builder = StateGraph(AdvancedRAGState)

    # ========== CONVERSATIONAL PREPROCESSING ==========
    builder.add_node("conversational_rewrite", conversational_rewrite_node)

    # ========== QUERY OPTIMIZATION STAGE ==========
    builder.add_node("query_expansion", query_expansion_node)
    builder.add_node("decide_strategy", decide_retrieval_strategy_node)

    # ========== RETRIEVAL STAGE ==========
    builder.add_node("retrieve_with_expansion", retrieve_with_expansion_node)
    builder.add_node("rewrite_and_refine", rewrite_and_refine_node)

    # ========== ANSWER STAGE ==========
    builder.add_node("answer_generation", answer_generation_with_quality_node)
    builder.add_node("groundedness_check", groundedness_check_node)
    builder.add_node("evaluate_answer", evaluate_answer_with_retrieval_node)

    # ========== EDGES ==========
    # Start with conversational rewrite, then query optimization
    builder.add_edge(START, "conversational_rewrite")
    builder.add_edge("conversational_rewrite", "query_expansion")

    # Conditional routing from query_expansion (RAG-Fusion pattern):
    # - Initial flow: generate strategy-agnostic expansions → decide best strategy
    # - Retry flow: strategy already changed → skip decide_strategy, go straight to retrieve
    builder.add_conditional_edges(
        "query_expansion",
        route_after_query_expansion,
        {
            "decide_strategy": "decide_strategy",
            "retrieve_with_expansion": "retrieve_with_expansion",
        }
    )

    builder.add_edge("decide_strategy", "retrieve_with_expansion")

    # Route based on retrieval quality (content-driven with early strategy switching)
    builder.add_conditional_edges(
        "retrieve_with_expansion",
        route_after_retrieval,
        {
            "answer_generation": "answer_generation",
            "rewrite_and_refine": "rewrite_and_refine",
            "query_expansion": "query_expansion",
        }
    )

    # Rewrite always routes to query expansion (expansions always cleared by rewrite_and_refine_node)
    # Critical: Query rewriting clears expansions, requiring regeneration for effectiveness
    builder.add_edge("rewrite_and_refine", "query_expansion")

    # Answer generation leads to groundedness check
    builder.add_edge("answer_generation", "groundedness_check")

    # Route after groundedness check
    builder.add_conditional_edges(
        "groundedness_check",
        route_after_groundedness,
        {
            "answer_generation": "answer_generation",
            "evaluate_answer": "evaluate_answer",
        }
    )

    # Self-correction: routes to query_expansion (strategy always changes in retry logic)
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "query_expansion": "query_expansion",
            END: END,
        }
    )

    # Compile with memory
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph

# Create the graph
advanced_rag_graph = build_advanced_rag_graph()
