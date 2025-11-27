"""
Basic RAG Graph (1 Feature) - Simplest Possible RAG.

Baseline implementation showing the absolute minimum viable RAG system.
Just semantic search + answer generation, no optimizations.

Features (1):
1. Semantic vector search

Graph Structure: 2 nodes, linear flow (no routing)
- START -> retrieve -> generate -> END

No query expansion - uses original query only.
No hybrid search - semantic/vector search only.
No RRF fusion - single query, single retrieval.
No reranking - directly uses top-k chunks.
No quality gates - assumes results are good enough.
No retry logic - single pass only.

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.utils.env import is_langgraph_api_environment
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


# ========== STATE SCHEMA ==========

class BasicRAGState(TypedDict):
    """Minimal state for basic RAG."""
    user_question: str
    retrieved_docs: list
    unique_docs_list: Optional[list]
    final_answer: Optional[str]
    confidence_score: Optional[float]
    ground_truth_doc_ids: Optional[list]


# ========== GLOBALS ==========

adaptive_retriever = None


# ========== NODES ==========

def retrieve_node(state: BasicRAGState) -> dict:
    """Basic semantic retrieval - top-k chunks, no reranking."""
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    query = state["user_question"]

    # Single semantic search, no RRF, no reranking
    k_final = adaptive_retriever.k_final if adaptive_retriever else 4
    docs = adaptive_retriever.retrieve_without_reranking(query, strategy="semantic", k_total=k_final)

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    print(f"\n{'='*60}")
    print(f"BASIC RETRIEVAL")
    print(f"Strategy: semantic only (vector similarity)")
    print(f"Top-K: {k_final} chunks (no reranking)")
    print(f"Retrieved: {len(docs)} documents")

    # Show ALL chunk IDs (rank order = quality indicator for basic retrieval)
    print(f"\nAll {len(docs)} chunk IDs (rank order):")
    for i, doc in enumerate(docs, 1):
        chunk_id = doc.metadata.get("id", "unknown")
        print(f"  {i}. {chunk_id}")

    # Show ground truth tracking
    if ground_truth_doc_ids:
        retrieved_chunk_ids = [doc.metadata.get("id", "unknown") for doc in docs]
        found_chunks = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in retrieved_chunk_ids]
        missing_chunks = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in retrieved_chunk_ids]
        print(f"\nExpected chunks: {ground_truth_doc_ids}")
        print(f"Found: {found_chunks if found_chunks else '[]'} | Missing: {missing_chunks if missing_chunks else '[]'}")

    print(f"{'='*60}\n")

    return {
        "retrieved_docs": docs,
        "unique_docs_list": docs,
    }


def generate_node(state: BasicRAGState) -> dict:
    """Simple answer generation without quality assessment."""
    query = state["user_question"]
    docs = state.get("retrieved_docs", [])

    # Generate answer
    spec = get_model_for_task("answer_generation")
    model_kwargs = {}
    if spec.reasoning_effort:
        model_kwargs["reasoning_effort"] = spec.reasoning_effort
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        model_kwargs=model_kwargs
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question using the provided context.

Context:
{context}

Question: {query}

Answer:"""

    answer = llm.invoke(prompt).content

    print(f"\n{'='*60}")
    print(f"ANSWER GENERATION")
    print(f"Answer length: {len(answer)} chars")
    print(f"Context docs: {len(docs)}")
    print(f"{'='*60}\n")

    # Fixed confidence (no assessment)
    confidence = 0.5  # Lower baseline confidence

    return {
        "final_answer": answer,
        "confidence_score": confidence,
    }


# ========== GRAPH BUILDER ==========

def build_basic_rag_graph():
    """Build simplest possible RAG graph."""
    builder = StateGraph(BasicRAGState)

    # Add nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Linear flow (no conditional routing)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # Skip checkpointer when running under LangGraph API (provides its own persistence)
    checkpointer = None if is_langgraph_api_environment() else MemorySaver()
    return builder.compile(checkpointer=checkpointer)


basic_rag_graph = build_basic_rag_graph()
