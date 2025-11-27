"""
Intermediate RAG Graph (5 Features) - Simple Linear RAG Pipeline.

Demonstrates core RAG functionality without advanced routing or retry logic.
Linear flow: query expansion -> retrieval -> reranking -> answer generation.

Features (5 = +4 over Basic):
1. Semantic vector search (inherited)
2. Query expansion (multi-variant)
3. Hybrid retrieval (semantic + BM25)
4. RRF fusion
5. CrossEncoder reranking

Graph Structure: 4 nodes, linear flow (no routing)
- START -> query_expansion -> retrieve -> rerank -> generate -> END

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
from advanced_agentic_rag_langgraph.utils.env import is_langgraph_api_environment
from advanced_agentic_rag_langgraph.retrieval import expand_query
from advanced_agentic_rag_langgraph.retrieval.cross_encoder_reranker import CrossEncoderReRanker


# ========== STATE SCHEMA ==========

class IntermediateRAGState(TypedDict):
    """Minimal state for intermediate RAG."""
    user_question: str
    query_expansions: list[str]
    retrieved_docs: Annotated[list[str], operator.add]
    unique_docs_list: Optional[list]
    final_answer: Optional[str]
    confidence_score: Optional[float]
    ground_truth_doc_ids: Optional[list]


# ========== GLOBALS ==========

adaptive_retriever = None
cross_encoder = CrossEncoderReRanker()


# ========== NODES ==========

def query_expansion_node(state: IntermediateRAGState) -> dict:
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


def retrieve_node(state: IntermediateRAGState) -> dict:
    """Hybrid retrieval with RRF fusion."""
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    query = state["user_question"]
    expansions = state.get("query_expansions", [])

    # RRF fusion implementation (inline, as in nodes.py)
    doc_ranks = {}
    doc_objects = {}

    for q in expansions:
        docs = adaptive_retriever.retrieve_without_reranking(q, strategy="hybrid")

        for rank, doc in enumerate(docs, start=1):
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

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL WITH RRF FUSION")
    print(f"Strategy: hybrid (always)")
    print(f"Query variants: {len(expansions)}")
    print(f"Total retrievals: {sum(len(ranks) for ranks in doc_ranks.values())}")
    print(f"Unique docs after RRF: {len(unique_docs)}")

    # Show ALL chunk IDs with RRF scores
    print(f"\nAll {len(sorted_doc_ids)} chunk IDs (RRF scores):")
    for i, doc_id in enumerate(sorted_doc_ids, 1):
        print(f"  {i}. {doc_id} ({rrf_scores[doc_id]:.4f})")

    # Show ground truth tracking
    if ground_truth_doc_ids:
        found_chunks = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in sorted_doc_ids]
        missing_chunks = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in sorted_doc_ids]
        print(f"\nExpected chunks: {ground_truth_doc_ids}")
        print(f"Found: {found_chunks if found_chunks else '[]'} | Missing: {missing_chunks if missing_chunks else '[]'}")

    print(f"{'='*60}\n")

    return {
        "retrieved_docs": [doc.page_content for doc in unique_docs],
        "unique_docs_list": unique_docs,
    }


def rerank_node(state: IntermediateRAGState) -> dict:
    """CrossEncoder reranking only (stage 1, top-k)."""
    global adaptive_retriever
    query = state["user_question"]
    docs = state.get("unique_docs_list", [])

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    # Single-stage reranking with CrossEncoder
    ranked_results = cross_encoder.rank(query, docs[:15])
    k_final = adaptive_retriever.k_final if adaptive_retriever else 4
    reranked_docs = [doc for doc, score in ranked_results[:k_final]]

    print(f"\n{'='*60}")
    print(f"CROSSENCODER RERANKING")
    print(f"Input: {len(docs)} documents")

    # Show chunk IDs going into reranking
    reranking_chunk_ids = [doc.metadata.get("id", "unknown") for doc in docs[:15]]
    print(f"\nChunk IDs sent to reranking (top-15):")
    for i, chunk_id in enumerate(reranking_chunk_ids[:10], 1):
        print(f"  {i}. {chunk_id}")
    if len(reranking_chunk_ids) > 10:
        print(f"  ... and {len(reranking_chunk_ids) - 10} more")

    # Track ground truth in reranking input
    if ground_truth_doc_ids:
        found_in_reranking = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in reranking_chunk_ids]
        missing_in_reranking = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in reranking_chunk_ids]
        print(f"\nExpected chunks in reranking input:")
        print(f"Found: {found_in_reranking if found_in_reranking else '[]'} | Missing: {missing_in_reranking if missing_in_reranking else '[]'}")

    print(f"\nOutput: {len(reranked_docs)} documents after CrossEncoder reranking")

    # Show final chunk IDs with reranking scores
    print(f"\nFinal chunk IDs (after CrossEncoder reranking):")
    for i, (doc, score) in enumerate(ranked_results[:k_final], 1):
        chunk_id = doc.metadata.get("id", "unknown")
        print(f"  {i}. {chunk_id} (score: {score:.4f})")

    # Track ground truth in final results
    if ground_truth_doc_ids:
        final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in reranked_docs]
        found_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in final_chunk_ids]
        missing_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in final_chunk_ids]
        print(f"\nExpected chunks in final results:")
        print(f"Found: {found_in_final if found_in_final else '[]'} | Missing: {missing_in_final if missing_in_final else '[]'}")

    print(f"{'='*60}\n")

    return {"unique_docs_list": reranked_docs}


def generate_node(state: IntermediateRAGState) -> dict:
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

def build_intermediate_rag_graph():
    """Build intermediate linear RAG graph."""
    builder = StateGraph(IntermediateRAGState)

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

    # Skip checkpointer when running under LangGraph API (provides its own persistence)
    checkpointer = None if is_langgraph_api_environment() else MemorySaver()
    return builder.compile(checkpointer=checkpointer)


intermediate_rag_graph = build_intermediate_rag_graph()
