from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.core import AdvancedRAGState
from src.orchestration.nodes import (
    query_expansion_node,
    decide_retrieval_strategy_node,
    retrieve_with_expansion_node,
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
    """Route based on answer evaluation - try different strategy if needed"""
    if state.get("is_answer_sufficient"):
        return END
    elif state.get("retrieval_attempts", 0) < 3:  # Max 3 attempts
        # Try different retrieval strategy
        current = state.get("retrieval_strategy", "hybrid")
        if current == "hybrid":
            state["retrieval_strategy"] = "semantic"
        elif current == "semantic":
            state["retrieval_strategy"] = "keyword"
        else:
            return END  # Give up
        return "retrieve_with_expansion"
    else:
        return END

def build_advanced_rag_graph():
    """Build complete advanced RAG graph with all techniques"""
    builder = StateGraph(AdvancedRAGState)

    # ========== QUERY OPTIMIZATION STAGE ==========
    builder.add_node("query_expansion", query_expansion_node)
    builder.add_node("decide_strategy", decide_retrieval_strategy_node)

    # ========== RETRIEVAL STAGE ==========
    builder.add_node("retrieve_with_expansion", retrieve_with_expansion_node)
    builder.add_node("rewrite_and_refine", rewrite_and_refine_node)

    # ========== ANSWER STAGE ==========
    builder.add_node("answer_generation", answer_generation_with_quality_node)
    builder.add_node("evaluate_answer", evaluate_answer_with_retrieval_node)

    # ========== EDGES ==========
    # Start with query optimization
    builder.add_edge(START, "query_expansion")
    builder.add_edge("query_expansion", "decide_strategy")
    builder.add_edge("decide_strategy", "retrieve_with_expansion")

    # Route based on retrieval quality
    builder.add_conditional_edges(
        "retrieve_with_expansion",
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
            "END": END,
        }
    )

    # Compile with memory
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph

# Create the graph
advanced_rag_graph = build_advanced_rag_graph()
