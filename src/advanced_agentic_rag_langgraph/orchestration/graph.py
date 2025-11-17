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
from typing import Literal

def route_after_retrieval(state: AdvancedRAGState) -> Literal["answer_generation", "rewrite_and_refine", "query_expansion"]:
    """
    Route based on retrieval quality with content-driven early strategy switching.

    Implements dual-tier strategy switching:
    - Early tier (here): Detects obvious strategy mismatches (off_topic, wrong_domain)
    - Late tier (route_after_evaluation): Handles subtle insufficiency after answer generation

    Research-backed CRAG pattern: confidence-based action triggering.
    """
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])
    current_strategy = state.get("retrieval_strategy", "hybrid")

    # Good quality: proceed to answer generation
    if quality > 0.6:
        return "answer_generation"

    # Max attempts reached: forced to proceed
    if attempts >= 2:
        return "answer_generation"

    # Poor quality: Content-driven decision
    # Strategy mismatch indicators suggest fundamental approach problem, not query problem
    if "off_topic" in issues or "wrong_domain" in issues:
        # Switch strategy instead of rewriting query
        # This avoids wasting retrieval attempts on wrong strategy
        next_strategy = select_next_strategy(current_strategy, issues)
        state["retrieval_strategy"] = next_strategy
        state["strategy_switch_reason"] = f"Early detection: {', '.join(issues)}"
        state["strategy_changed"] = True
        state["query_expansions"] = None  # Trigger regeneration for new strategy

        print(f"\n{'='*60}")
        print(f"EARLY STRATEGY SWITCH")
        print(f"From: {current_strategy} to {next_strategy}")
        print(f"Reason: {', '.join(issues)}")
        print(f"Attempt: {attempts + 1}")
        print(f"Quality score: {quality:.0%}")
        print(f"{'='*60}\n")

        return "query_expansion"  # Regenerate expansions for new strategy
    else:
        # Other issues (partial_coverage, missing_key_info, etc.) may benefit from query rewriting
        return "rewrite_and_refine"

def route_after_groundedness(state: AdvancedRAGState) -> Literal["answer_generation", "evaluate_answer"]:
    """
    Route based on groundedness with retrieval quality awareness and false positive protection.

    Three-tier threshold strategy:
    - NONE (0.8-1.0): No hallucination, proceed
    - MODERATE (0.6-0.8): Likely NLI false positive, proceed without retry
    - SEVERE (<0.6): Root cause detection:
      * Good retrieval + low groundedness → LLM hallucination (retry generation)
      * Poor retrieval + low groundedness → Retrieval-caused (flag for re-retrieval)

    Research-backed approach:
    - Protects against over-conservative NLI detector (zero-shot F1: 0.65-0.70)
    - Re-retrieval reduces hallucination 46% more than regeneration when context is the issue
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
            print(f"\n{'='*60}")
            print(f"GROUNDEDNESS ISSUE (Retrieval-Caused)")
            print(f"Groundedness: {groundedness_score:.0%}")
            print(f"Retrieval quality: {retrieval_quality:.0%} (POOR)")
            print(f"Root cause: LLM filled gaps due to insufficient context")
            print(f"Action: Flagging for re-retrieval (not regeneration)")
            print(f"Research: Re-retrieval > regeneration for context gaps")
            print(f"{'='*60}\n")

            # Flag for evaluation to trigger re-retrieval with strategy change
            state["retrieval_caused_hallucination"] = True
            state["is_answer_sufficient"] = False  # Force retry in evaluation

            return "evaluate_answer"

    # Default: proceed to evaluation
    return "evaluate_answer"


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


def route_after_evaluation(state: AdvancedRAGState) -> Literal["retrieve_with_expansion", "query_expansion", "END"]:
    """
    Route based on answer evaluation with content-driven strategy switching.

    Uses retrieval quality issues (content-based) to intelligently select
    next retrieval strategy, following research-backed CRAG/Self-RAG patterns.

    Priority checks:
    1. Retrieval-caused hallucination → Force re-retrieval with strategy change
    2. Answer sufficient → END
    3. Answer insufficient → Content-driven strategy switching
    """
    # PRIORITY CHECK: Retrieval-caused hallucination detected
    # Research: Re-retrieval reduces hallucination 46% more than regeneration
    if state.get("retrieval_caused_hallucination"):
        retrieval_attempts = state.get("retrieval_attempts", 0)

        if retrieval_attempts < 3:
            # Force re-retrieval with strategy change to address context gaps
            current_strategy = state.get("retrieval_strategy", "hybrid")

            # Intelligent strategy switching based on current strategy
            if current_strategy == "semantic":
                next_strategy = "keyword"  # Try precision-based approach
            elif current_strategy == "keyword":
                next_strategy = "hybrid"   # Try balanced approach
            else:  # hybrid
                next_strategy = "semantic"  # Try conceptual approach

            print(f"\n{'='*60}")
            print(f"RE-RETRIEVAL (Hallucination Mitigation)")
            print(f"Trigger: Poor retrieval caused hallucination")
            print(f"Strategy: {current_strategy} to {next_strategy}")
            print(f"Attempt: {retrieval_attempts + 1}/3")
            print(f"Research: Re-retrieval > regeneration for context gaps")
            print(f"{'='*60}\n")

            # Update state for strategy change
            state["retrieval_strategy"] = next_strategy
            state["strategy_changed"] = True
            state["query_expansions"] = None  # Regenerate for new strategy
            state["retrieval_caused_hallucination"] = False  # Clear flag

            return "query_expansion"  # Regenerate expansions for new strategy
        else:
            # Max attempts reached, give up
            return END

    # Standard evaluation flow
    if state.get("is_answer_sufficient"):
        return END
    elif state.get("retrieval_attempts", 0) < 3:  # Max 3 attempts
        # Content-driven strategy selection based on retrieval quality issues
        retrieval_quality_issues = state.get("retrieval_quality_issues", [])
        retrieval_quality_score = state.get("retrieval_quality_score", 0.7)
        current = state.get("retrieval_strategy", "hybrid")

        # Map content issues to optimal strategies (research-backed)
        if "missing_key_info" in retrieval_quality_issues and retrieval_quality_score < 0.6:
            # Missing information suggests need for semantic/conceptual search
            if current != "semantic":
                next_strategy = "semantic"
                reasoning = f"Content-driven: Missing key information detected, switching to semantic search for better conceptual coverage"
            else:
                # Already semantic, try hybrid for balanced approach
                next_strategy = "hybrid"
                reasoning = f"Content-driven: Semantic failed to find key information, trying hybrid for broader coverage"
        elif "off_topic" in retrieval_quality_issues or "wrong_domain" in retrieval_quality_issues:
            # Off-topic results suggest need for precise keyword matching
            if current != "keyword":
                next_strategy = "keyword"
                reasoning = f"Content-driven: Off-topic results detected, switching to keyword search for precision"
            else:
                # Already keyword, try hybrid
                next_strategy = "hybrid"
                reasoning = f"Content-driven: Keyword search not precise enough, trying hybrid"
        elif "partial_coverage" in retrieval_quality_issues or "incomplete_context" in retrieval_quality_issues:
            # Partial coverage suggests need for different search approach
            if current == "hybrid":
                next_strategy = "semantic"
                reasoning = "Content-driven: Partial coverage with hybrid, trying semantic for depth"
            elif current == "semantic":
                next_strategy = "keyword"
                reasoning = "Content-driven: Semantic incomplete, trying keyword for specificity"
            else:
                next_strategy = "hybrid"
                reasoning = "Content-driven: Keyword insufficient, trying hybrid for balance"
        else:
            # Fallback: traditional progression (hybrid → semantic → keyword)
            if current == "hybrid":
                next_strategy = "semantic"
                reasoning = "Fallback: hybrid to semantic"
            elif current == "semantic":
                next_strategy = "keyword"
                reasoning = "Fallback: semantic to keyword"
            else:
                return END  # Exhausted all strategies

        # Log refinement
        refinement = {
            "iteration": state.get("retrieval_attempts", 0),
            "from_strategy": current,
            "to_strategy": next_strategy,
            "reasoning": reasoning,
            "retrieval_quality_issues": retrieval_quality_issues,
            "retrieval_quality_score": retrieval_quality_score,
        }

        # Check if strategy changed - if yes, need to regenerate query expansions
        strategy_changed = (next_strategy != current)

        print(f"\n{'='*60}")
        print(f"STRATEGY REFINEMENT")
        print(f"Iteration: {refinement['iteration']}")
        print(f"Switch: {current} to {next_strategy}")
        print(f"Reasoning: {reasoning}")
        print(f"Retrieval quality: {retrieval_quality_score:.0%}")
        print(f"Detected issues: {', '.join(retrieval_quality_issues) if retrieval_quality_issues else 'None'}")
        if strategy_changed:
            print(f"Strategy changed: Will regenerate query expansions for new strategy")
        print(f"{'='*60}\n")

        # Update state with new strategy and log refinement
        state["retrieval_strategy"] = next_strategy
        state.setdefault("refinement_history", []).append(refinement)

        # If strategy changed, clear expansions and route through query_expansion
        if strategy_changed:
            state["query_expansions"] = None  # Signal regeneration needed
            state["strategy_changed"] = True  # Flag for logging
            return "query_expansion"  # Regenerate expansions for new strategy
        else:
            # Same strategy, can reuse existing expansions
            return "retrieve_with_expansion"
    else:
        return END

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

    # Conditional routing from query_expansion:
    # - If called from initial flow: go to decide_strategy
    # - If called from retry loop (strategy_changed): go directly to retrieve_with_expansion
    def route_after_query_expansion(state: AdvancedRAGState) -> Literal["decide_strategy", "retrieve_with_expansion"]:
        """Route based on whether this is initial expansion or retry with strategy change"""
        if state.get("strategy_changed", False):
            # Strategy changed, skip decide_strategy (already set in route_after_evaluation)
            return "retrieve_with_expansion"
        else:
            # Initial flow, proceed to strategy selection
            return "decide_strategy"

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
            "query_expansion": "query_expansion",  # Early strategy switch
        }
    )

    # Rewrite loops back to retrieval with new query
    builder.add_edge("rewrite_and_refine", "retrieve_with_expansion")

    # Answer generation leads to groundedness check
    builder.add_edge("answer_generation", "groundedness_check")

    # Route after groundedness check
    builder.add_conditional_edges(
        "groundedness_check",
        route_after_groundedness,
        {
            "answer_generation": "answer_generation",  # Retry if severe hallucination
            "evaluate_answer": "evaluate_answer",      # Proceed to evaluation
        }
    )

    # Self-correction: try different strategy if answer insufficient
    # Now supports routing to query_expansion when strategy changes (for regeneration)
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "retrieve_with_expansion": "retrieve_with_expansion",
            "query_expansion": "query_expansion",  # When strategy changes, regenerate expansions
            END: END,
        }
    )

    # Compile with memory
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph

# Create the graph
advanced_rag_graph = build_advanced_rag_graph()
