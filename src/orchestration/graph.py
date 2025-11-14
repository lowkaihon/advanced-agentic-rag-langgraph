from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.core import AdvancedRAGState
from src.orchestration.nodes import (
    conversational_rewrite_node,
    query_expansion_node,
    decide_retrieval_strategy_node,
    retrieve_with_expansion_node,
    analyze_retrieved_metadata_node,
    rewrite_and_refine_node,
    answer_generation_with_quality_node,
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

def route_after_evaluation(state: AdvancedRAGState) -> Literal["retrieve_with_expansion", "END"]:
    """
    Route based on answer evaluation with metadata-driven strategy switching.

    Uses document metadata analysis to intelligently select next strategy.
    """
    if state.get("is_answer_sufficient"):
        return END
    elif state.get("retrieval_attempts", 0) < 3:  # Max 3 attempts
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

        # Use metadata suggestion if available, otherwise use fallback order
        if suggested_strategy and suggested_strategy != current:
            next_strategy = suggested_strategy
            reasoning = f"Metadata-driven: switching to {suggested_strategy} (current: {current})"
        else:
            # Fallback to traditional order: hybrid → semantic → keyword
            if current == "hybrid":
                next_strategy = "semantic"
                reasoning = "Fallback: hybrid → semantic"
            elif current == "semantic":
                next_strategy = "keyword"
                reasoning = "Fallback: semantic → keyword"
            else:
                return END  # Give up

        # Log refinement
        refinement = {
            "iteration": state.get("retrieval_attempts", 0),
            "from_strategy": current,
            "to_strategy": next_strategy,
            "reasoning": reasoning,
            "metadata_issues": [issue.get("issue") for issue in quality_issues]
        }

        print(f"\n{'='*60}")
        print(f"STRATEGY REFINEMENT")
        print(f"Iteration: {refinement['iteration']}")
        print(f"Switch: {current} → {next_strategy}")
        print(f"Reasoning: {reasoning}")
        print(f"Metadata issues: {refinement['metadata_issues']}")
        print(f"{'='*60}\n")

        # Update state with new strategy and log refinement
        state["retrieval_strategy"] = next_strategy
        state.setdefault("refinement_history", []).append(refinement)

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
    builder.add_node("evaluate_answer", evaluate_answer_with_retrieval_node)

    # ========== EDGES ==========
    # Start with conversational rewrite, then query optimization
    builder.add_edge(START, "conversational_rewrite")
    builder.add_edge("conversational_rewrite", "query_expansion")
    builder.add_edge("query_expansion", "decide_strategy")
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

    # Answer generation leads to evaluation
    builder.add_edge("answer_generation", "evaluate_answer")

    # Self-correction: try different strategy if answer insufficient
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "retrieve_with_expansion": "retrieve_with_expansion",
            END: END,
        }
    )

    # Compile with memory
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph

# Create the graph
advanced_rag_graph = build_advanced_rag_graph()
