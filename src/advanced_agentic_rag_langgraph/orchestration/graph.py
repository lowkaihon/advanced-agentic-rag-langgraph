from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from advanced_agentic_rag_langgraph.core import AdvancedRAGState
from advanced_agentic_rag_langgraph.orchestration.nodes import (
    conversational_rewrite_node,
    query_expansion_node,
    decide_retrieval_strategy_node,
    retrieve_with_expansion_node,
    analyze_retrieved_metadata_node,
    rewrite_and_refine_node,
    answer_generation_with_quality_node,
    groundedness_check_node,
    evaluate_answer_with_retrieval_node,
)
from typing import Literal

def route_after_retrieval(state: AdvancedRAGState) -> Literal["answer_generation", "rewrite_and_refine"]:
    """Route based on retrieval quality"""
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)

    # If quality is good OR we've tried rewrites twice, proceed to answer
    if quality > 0.6 or attempts >= 2:
        return "answer_generation"
    else:
        return "rewrite_and_refine"

def route_after_groundedness(state: AdvancedRAGState) -> Literal["answer_generation", "evaluate_answer"]:
    """
    Route based on groundedness check results.

    Conditional blocking strategy (best practice):
    - retry_needed (score < 0.6) AND retry_count < 1: Regenerate with stricter prompt
    - Else: Proceed to evaluation (with warnings if hallucination detected)
    """
    retry_needed = state.get("retry_needed", False)
    retry_count = state.get("groundedness_retry_count", 0)

    if retry_needed and retry_count < 1:
        # Severe hallucination: retry generation once
        print(f"\n{'='*60}")
        print(f"GROUNDEDNESS RETRY")
        print(f"Retry count: {retry_count}/1")
        print(f"Action: Regenerating answer with stricter grounding instructions")
        print(f"{'='*60}\n")
        return "answer_generation"
    else:
        # Proceed to evaluation
        # If hallucination was detected but not severe enough to retry,
        # or if we already retried once, continue with warning
        return "evaluate_answer"


def route_after_evaluation(state: AdvancedRAGState) -> Literal["retrieve_with_expansion", "query_expansion", "END"]:
    """
    Route based on answer evaluation with metadata-driven strategy switching.

    Streamlined: Uses retrieval_quality_issues to detect missing information,
    eliminating redundant context sufficiency check.
    """
    if state.get("is_answer_sufficient"):
        return END
    elif state.get("retrieval_attempts", 0) < 3:  # Max 3 attempts
        # Check for missing information from retrieval quality issues
        retrieval_quality_issues = state.get("retrieval_quality_issues", [])
        has_missing_info = any(issue in retrieval_quality_issues for issue in ["partial_coverage", "missing_key_info", "incomplete_context"])
        retrieval_quality_score = state.get("retrieval_quality_score", 0.7)

        # Metadata-driven strategy switching
        current = state.get("retrieval_strategy", "hybrid")
        metadata_analysis = state.get("doc_metadata_analysis", {})
        quality_issues = metadata_analysis.get("quality_issues", [])

        # Check if metadata analysis suggests a specific strategy
        suggested_strategy = None
        for issue in quality_issues:
            if "suggested_strategy" in issue:
                suggested_strategy = issue["suggested_strategy"]
                break

        # If retrieval detected missing information, prioritize semantic search
        # (semantic may capture conceptual completeness better than keyword)
        if has_missing_info and retrieval_quality_score < 0.6:
            if current != "semantic":
                next_strategy = "semantic"
                reasoning = f"Quality-driven: Missing information detected ({', '.join([i for i in retrieval_quality_issues if i in ['partial_coverage', 'missing_key_info', 'incomplete_context']])}), switching to semantic search"
            else:
                # Already semantic, try hybrid as fallback
                next_strategy = "hybrid"
                reasoning = f"Quality-driven: Semantic failed to provide complete information, trying hybrid"
        # Use metadata suggestion if available and no context issues
        elif suggested_strategy and suggested_strategy != current:
            next_strategy = suggested_strategy
            reasoning = f"Metadata-driven: switching to {suggested_strategy} (current: {current})"
        else:
            # Fallback to traditional order: hybrid to semantic to keyword
            if current == "hybrid":
                next_strategy = "semantic"
                reasoning = "Fallback: hybrid to semantic"
            elif current == "semantic":
                next_strategy = "keyword"
                reasoning = "Fallback: semantic to keyword"
            else:
                return END  # Give up

        # Log refinement
        refinement = {
            "iteration": state.get("retrieval_attempts", 0),
            "from_strategy": current,
            "to_strategy": next_strategy,
            "reasoning": reasoning,
            "metadata_issues": [issue.get("issue") for issue in quality_issues],
            "retrieval_quality_issues": retrieval_quality_issues,  # Track retrieval quality issues
            "retrieval_quality_score": retrieval_quality_score,  # Track retrieval quality score
        }

        # Check if strategy changed - if yes, need to regenerate query expansions
        strategy_changed = (next_strategy != current)

        print(f"\n{'='*60}")
        print(f"STRATEGY REFINEMENT")
        print(f"Iteration: {refinement['iteration']}")
        print(f"Switch: {current} to {next_strategy}")
        print(f"Reasoning: {reasoning}")
        print(f"Metadata issues: {refinement['metadata_issues']}")
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
    builder.add_node("analyze_metadata", analyze_retrieved_metadata_node)
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

    # After retrieval, analyze metadata before routing
    builder.add_edge("retrieve_with_expansion", "analyze_metadata")

    # Route based on retrieval quality (now from metadata analysis)
    builder.add_conditional_edges(
        "analyze_metadata",
        route_after_retrieval,
        {
            "answer_generation": "answer_generation",
            "rewrite_and_refine": "rewrite_and_refine",
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
