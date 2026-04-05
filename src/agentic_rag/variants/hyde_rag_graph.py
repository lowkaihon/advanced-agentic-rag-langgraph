"""HyDE RAG (2 features): Hypothetical Document Embeddings + semantic search.

Generates a hypothetical answer document before retrieval to improve semantic matching.
Linear flow with one additional node compared to basic RAG.

Graph: START -> hyde_generate -> retrieve -> generate -> END
"""

from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.utils.env import is_langgraph_api_environment
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


# ========== STATE SCHEMA ==========

class HyDERAGState(TypedDict):
    """State for HyDE RAG - adds hypothetical_doc to basic state."""
    user_question: str
    hypothetical_doc: Optional[str]
    retrieved_docs: list
    unique_docs_list: Optional[list]
    final_answer: Optional[str]
    confidence_score: Optional[float]
    ground_truth_doc_ids: Optional[list]


# ========== GLOBALS ==========

adaptive_retriever = None


# ========== PROMPTS ==========

HYDE_SYSTEM_PROMPT = """You are a HyDE (Hypothetical Document Embeddings) generator.
Your task is to generate a hypothetical document that would answer the user's question.

Rules:
1. Generate a detailed, informative passage (150-250 words)
2. Include specific terms, technical vocabulary, and concrete details
3. Do NOT use conversational language ("Sure", "Here is", "I think")
4. Do NOT answer the user directly - generate a document that CONTAINS the answer
5. Focus on keywords and semantic patterns that would appear in relevant documents
6. If the question is ambiguous, cover the most likely interpretation"""


# ========== NODES ==========

def hyde_generate_node(state: HyDERAGState) -> dict:
    """Generate a hypothetical document that would answer the query.

    This document is used for embedding-based retrieval instead of the raw query,
    improving semantic matching by shifting from query-to-document similarity
    to answer-to-document similarity.
    """
    query = state["user_question"]

    # Use higher temperature for creative keyword generation
    spec = get_model_for_task("query_expansion")  # Reuse expansion task config
    llm = ChatOpenAI(
        model=spec.name,
        temperature=0.7,  # HyDE benefits from creative generation
        max_tokens=300,   # Respect embedding window limit
    )

    messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nHypothetical Document:"}
    ]

    hypothetical_doc = llm.invoke(messages).content

    print(f"\n{'='*60}")
    print(f"HYDE GENERATION")
    print(f"Original query: {query[:80]}...")
    print(f"Hypothetical doc length: {len(hypothetical_doc)} chars")
    print(f"Hypothetical doc preview: {hypothetical_doc[:150]}...")
    print(f"{'='*60}\n")

    return {"hypothetical_doc": hypothetical_doc}


def retrieve_node(state: HyDERAGState) -> dict:
    """Retrieve using hypothetical document embedding instead of raw query.

    Key difference from basic RAG: Uses the generated hypothetical document
    for semantic search, not the original user question.
    """
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    # Use hypothetical doc for retrieval (core HyDE technique)
    search_text = state.get("hypothetical_doc") or state["user_question"]
    original_query = state["user_question"]

    # Single semantic search, no RRF, no reranking
    k_final = adaptive_retriever.k_final if adaptive_retriever else 4
    docs = adaptive_retriever.retrieve_without_reranking(search_text, strategy="semantic", k_total=k_final)

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    print(f"\n{'='*60}")
    print(f"HYDE RETRIEVAL")
    print(f"Strategy: semantic search using hypothetical document")
    print(f"Top-K: {k_final} chunks (no reranking)")
    print(f"Retrieved: {len(docs)} documents")

    # Show ALL chunk IDs (rank order = quality indicator)
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


def generate_node(state: HyDERAGState) -> dict:
    """Generate answer using original query (not hypothetical doc).

    Important: Answer generation uses the original user_question to maintain
    proper grounding to the user's actual intent.
    """
    query = state["user_question"]  # Use original query, not hypothetical doc
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

def build_hyde_rag_graph():
    """Build HyDE RAG graph with hypothetical document generation."""
    builder = StateGraph(HyDERAGState)

    # Add nodes
    builder.add_node("hyde_generate", hyde_generate_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Linear flow: hyde_generate -> retrieve -> generate
    builder.add_edge(START, "hyde_generate")
    builder.add_edge("hyde_generate", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # Skip checkpointer when running under LangGraph API (provides its own persistence)
    checkpointer = None if is_langgraph_api_environment() else MemorySaver()
    return builder.compile(checkpointer=checkpointer)


hyde_rag_graph = build_hyde_rag_graph()
