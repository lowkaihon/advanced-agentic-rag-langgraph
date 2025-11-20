"""
Pure Semantic RAG Graph (4 Features) - Simplest Possible RAG.

Baseline implementation showing the absolute minimum viable RAG system.
Just semantic search + answer generation, no optimizations.

Features (4):
1. Pure semantic search (vector similarity only)
2. Top-4 retrieval (no reranking)
3. Simple answer generation
4. Basic state management (TypedDict)

Graph Structure: 2 nodes, linear flow
- START → retrieve → generate → END

No query expansion - uses original query only.
No hybrid search - semantic/vector search only.
No RRF fusion - single query, single retrieval.
No reranking - directly uses top 4 chunks.
No quality gates - assumes results are good enough.
No retry logic - single pass only.

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


# ========== STATE SCHEMA ==========

class PureSemanticRAGState(TypedDict):
    """Minimal state for pure semantic RAG."""
    user_question: str
    retrieved_docs: list
    final_answer: Optional[str]
    confidence_score: Optional[float]


# ========== GLOBALS ==========

adaptive_retriever = None


# ========== NODES ==========

def retrieve_node(state: PureSemanticRAGState) -> dict:
    """Pure semantic retrieval - top 4 chunks, no reranking."""
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    query = state["user_question"]

    # Single semantic search, no RRF, no reranking
    docs = adaptive_retriever.retrieve_without_reranking(query, strategy="semantic", k_total=4)

    print(f"\n{'='*60}")
    print(f"PURE SEMANTIC RETRIEVAL")
    print(f"Strategy: semantic only (vector similarity)")
    print(f"Top-K: 4 chunks (no reranking)")
    print(f"Retrieved: {len(docs)} documents")
    print(f"{'='*60}\n")

    return {"retrieved_docs": docs}


def generate_node(state: PureSemanticRAGState) -> dict:
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

def build_pure_semantic_rag_graph():
    """Build simplest possible RAG graph."""
    builder = StateGraph(PureSemanticRAGState)

    # Add nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Linear flow (no conditional routing)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


pure_semantic_rag_graph = build_pure_semantic_rag_graph()
