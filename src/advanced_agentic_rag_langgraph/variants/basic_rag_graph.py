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
from advanced_agentic_rag_langgraph.retrieval import expand_query
from advanced_agentic_rag_langgraph.retrieval.cross_encoder_reranker import CrossEncoderReRanker


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

    # Always expand (no LLM decision) - expand_query returns [original] + variations
    expansions = expand_query(query)

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
        adaptive_retriever = setup_retriever(force_new=False)

    query = state["user_question"]
    expansions = state.get("query_expansions", [])

    # RRF fusion implementation (inline, as in nodes.py)
    doc_ranks = {}
    doc_objects = {}

    for q in expansions:
        docs = adaptive_retriever.retrieve_without_reranking(q, strategy="hybrid")

        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
                doc_objects[doc_id] = doc
            doc_ranks[doc_id].append(rank)

    # RRF scoring
    k = 60
    rrf_scores = {}
    for doc_id, ranks in doc_ranks.items():
        rrf_score = sum(1.0 / (rank + k) for rank in ranks)
        rrf_scores[doc_id] = rrf_score

    # Sort by RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    unique_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]

    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL WITH RRF FUSION")
    print(f"Strategy: hybrid (always)")
    print(f"Query variants: {len(expansions)}")
    print(f"Unique docs after RRF: {len(unique_docs)}")
    if sorted_doc_ids[:3]:
        print(f"Top 3 RRF scores: {[f'{rrf_scores[doc_id]:.4f}' for doc_id in sorted_doc_ids[:3]]}")
    print(f"{'='*60}\n")

    return {
        "retrieved_docs": [doc.page_content for doc in unique_docs],
        "unique_docs_list": unique_docs,
    }


def rerank_node(state: BasicRAGState) -> dict:
    """CrossEncoder reranking only (stage 1, top-5)."""
    query = state["user_question"]
    docs = state.get("unique_docs_list", [])

    # Single-stage reranking with CrossEncoder
    ranked_results = cross_encoder.rank(query, docs[:15])
    reranked_docs = [doc for doc, score in ranked_results[:5]]

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
