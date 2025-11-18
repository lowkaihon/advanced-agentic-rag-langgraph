"""
Basic RAG Graph (8 Features) - Simple Linear RAG Pipeline.

Demonstrates core RAG functionality without advanced routing or retry logic.
Linear flow: query expansion → retrieval → reranking → answer generation.

Features (8):
1. Hybrid retrieval (semantic + keyword combined)
2. Basic query expansion (always generate 3 variants)
3. RRF fusion for multiple query variants
4. CrossEncoder reranking (stage 1 only, top-5)
5. Basic answer generation with context
6. Simple state management (TypedDict)
7. Linear graph structure (no conditional routing)
8. Basic metrics tracking

Graph Structure: 4 nodes, linear flow (no routing functions)
- START → query_expansion → retrieve → rerank → generate → END

No retry logic - single pass only.
No quality gates - assumes all results are good enough.
No strategy selection - always uses hybrid search.

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

import operator
from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from advanced_agentic_rag_langgraph.retrieval import expand_query, AdaptiveRetriever
from advanced_agentic_rag_langgraph.retrieval.reranking import CrossEncoderReRanker


# ========== STATE SCHEMA ==========

class BasicRAGState(TypedDict):
    """Minimal state for basic RAG."""
    user_question: str
    query_expansions: list[str]
    retrieved_docs: Annotated[list[str], operator.add]
    unique_docs_list: Optional[list]
    final_answer: Optional[str]
    confidence_score: Optional[float]


# ========== GLOBALS ==========

adaptive_retriever = None
cross_encoder = CrossEncoderReRanker()


# ========== NODES ==========

def query_expansion_node(state: BasicRAGState) -> dict:
    """Always expand query into 3 variants."""
    query = state["user_question"]

    # Always expand (no LLM decision)
    expansions = expand_query(query, num_variations=3)

    print(f"\n{'='*60}")
    print(f"QUERY EXPANSION")
    print(f"Original: {query}")
    print(f"Generated {len(expansions)} variations")
    print(f"{'='*60}\n")

    return {"query_expansions": expansions}


def retrieve_node(state: BasicRAGState) -> dict:
    """Hybrid retrieval with RRF fusion."""
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = AdaptiveRetriever(
            vectorstore=setup_retriever(force_new=False),
            top_k=15
        )

    query = state["user_question"]
    expansions = state.get("query_expansions", [])

    # Retrieve using hybrid strategy for all queries
    all_docs = []
    for q in [query] + expansions:
        docs = adaptive_retriever.retrieve(q, strategy="hybrid")
        all_docs.extend(docs)

    # RRF fusion
    from advanced_agentic_rag_langgraph.retrieval.fusion import fuse_rankings
    fused_docs = fuse_rankings(all_docs, k=60)

    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL")
    print(f"Strategy: hybrid (always)")
    print(f"Queries: {len([query] + expansions)}")
    print(f"Retrieved: {len(fused_docs)} documents (after RRF fusion)")
    print(f"{'='*60}\n")

    unique_docs = list({doc.page_content: doc for doc in fused_docs}.values())

    return {
        "retrieved_docs": [doc.page_content for doc in unique_docs],
        "unique_docs_list": unique_docs,
    }


def rerank_node(state: BasicRAGState) -> dict:
    """CrossEncoder reranking only (stage 1, top-5)."""
    query = state["user_question"]
    docs = state.get("unique_docs_list", [])

    # Single-stage reranking with CrossEncoder
    reranked_docs = cross_encoder.rerank(query, docs, top_k=5)

    print(f"\n{'='*60}")
    print(f"CROSSENCODER RERANKING")
    print(f"Input: {len(docs)} documents")
    print(f"Output: {len(reranked_docs)} documents")
    print(f"{'='*60}\n")

    return {"unique_docs_list": reranked_docs}


def generate_node(state: BasicRAGState) -> dict:
    """Basic answer generation without quality assessment."""
    query = state["user_question"]
    docs = state.get("unique_docs_list", [])

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

    # Simple confidence (no quality assessment)
    confidence = 0.7  # Fixed confidence score

    return {
        "final_answer": answer,
        "confidence_score": confidence,
    }


# ========== GRAPH BUILDER ==========

def build_basic_rag_graph():
    """Build basic linear RAG graph."""
    builder = StateGraph(BasicRAGState)

    # Add nodes
    builder.add_node("query_expansion", query_expansion_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)

    # Linear flow (no conditional routing)
    builder.add_edge(START, "query_expansion")
    builder.add_edge("query_expansion", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


basic_rag_graph = build_basic_rag_graph()
