from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from advanced_agentic_rag_langgraph.core import AdvancedRAGState
from advanced_agentic_rag_langgraph.orchestration.nodes import (
    conversational_rewrite_node,
    query_expansion_node,
    decide_retrieval_strategy_node,
    retrieve_with_expansion_node,
    rewrite_and_refine_node,
    answer_generation_node,
    evaluate_answer_node,
)
from typing import Literal


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

    print(f"\n{'='*60}")
    print(f"ROUTER: AFTER RETRIEVAL")
    print(f"Quality: {quality:.0%} (threshold: >=60%)")
    print(f"Attempts: {attempts}/3")
    if issues:
        print(f"Issues: {', '.join(issues)}")

    if quality >= 0.6:
        print(f"Decision: answer_generation (quality acceptable)")
        print(f"{'='*60}\n")
        return "answer_generation"

    if attempts >= 3:
        print(f"Decision: answer_generation (max attempts reached)")
        print(f"{'='*60}\n")
        return "answer_generation"

    if ("off_topic" in issues or "wrong_domain" in issues) and (attempts == 1):
        print(f"Decision: query_expansion (early strategy switch on first poor attempt)")
        print(f"{'='*60}\n")
        return "query_expansion"
    else:
        print(f"Decision: rewrite_and_refine (semantic rewrite)")
        print(f"{'='*60}\n")
        return "rewrite_and_refine"


# ========== EVALUATION ROUTING ==========

def route_after_evaluation(state: AdvancedRAGState) -> Literal["answer_generation", "END"]:
    """
    Single routing decision: retry generation or end.

    By this point, retrieval already validated upstream (quality >= 0.6 or attempts >= 3).
    Therefore all answer issues = generation problems.

    Research principle: "Fix generation problems with generation strategies, not by retrieving more documents."
    """

    # Priority 1: Answer sufficient -> done
    if state.get("is_answer_sufficient"):
        print("\nRouting: END (answer sufficient)")
        return END

    # Priority 2: Generation retry budget
    generation_attempts = state.get("generation_attempts", 0)

    if generation_attempts < 3:
        print(f"\nRouting: answer_generation (attempt {generation_attempts + 1}/3)")
        return "answer_generation"
    else:
        print(f"\nRouting: END (max attempts reached)")
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
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)

    builder.add_edge(START, "conversational_rewrite")
    builder.add_edge("conversational_rewrite", "decide_strategy")
    builder.add_edge("decide_strategy", "query_expansion")
    builder.add_edge("query_expansion", "retrieve_with_expansion")

    builder.add_conditional_edges(
        "retrieve_with_expansion",
        route_after_retrieval,
        {
            "answer_generation": "answer_generation",
            "rewrite_and_refine": "rewrite_and_refine",
            "query_expansion": "query_expansion",
        }
    )

    builder.add_edge("rewrite_and_refine", "query_expansion")

    # Simplified flow: answer_generation -> evaluate_answer (single evaluation node)
    builder.add_edge("answer_generation", "evaluate_answer")

    # Single conditional routing: retry generation or end (no re-retrieval after generation)
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "answer_generation": "answer_generation",
            END: END,
        }
    )

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph

advanced_rag_graph = build_advanced_rag_graph()
