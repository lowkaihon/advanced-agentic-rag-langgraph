"""
Intermediate RAG Graph (18 Features) - Enhanced RAG with Quality Gates.

Demonstrates value of conditional routing and quality-driven retry logic.
Builds on basic RAG by adding strategy selection, two-stage reranking, and simple retry loops.

Features (18 total = 8 basic + 10 new):
BASIC (8):
- Hybrid retrieval (semantic + keyword)
- Basic query expansion
- RRF fusion for query variants
- CrossEncoder reranking
- Answer generation
- Simple state management
- Basic metrics tracking
- Linear graph flow

NEW (10):
- Conversational query rewriting
- LLM-based strategy selection (semantic/keyword/hybrid)
- LLM-based expansion decision
- Two-stage reranking (CrossEncoder → LLM-as-judge)
- Binary retrieval quality scoring (good/bad)
- Query rewriting loop (generic feedback, max 1 rewrite)
- Answer quality check (yes/no sufficient)
- Conditional routing (2 router functions)
- Limited retry logic (1 query rewrite, 2 retrieval attempts)
- Message accumulation for multi-turn

Graph Structure: 7 nodes, 2 routing functions
- conversational_rewrite → query_expansion → decide_strategy → retrieve → rerank → grade → generate
- route_after_retrieval: quality >=0.6 OR attempts >=2 → generate, else rewrite
- route_after_generation: sufficient → END, else rewrite (max 1)

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

import operator
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter
from advanced_agentic_rag_langgraph.retrieval import expand_query, rewrite_query
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
from advanced_agentic_rag_langgraph.retrieval.cross_encoder_reranker import CrossEncoderReRanker
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics


# ========== STATE SCHEMA ==========

class IntermediateRAGState(TypedDict):
    """State for intermediate RAG with quality tracking."""
    # Always set
    user_question: str
    baseline_query: str

    # Conditionally set
    active_query: Optional[str]
    query_expansions: Optional[list[str]]
    retrieval_strategy: Optional[Literal["semantic", "keyword", "hybrid"]]

    # Accumulated
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: Annotated[list[str], operator.add]
    unique_docs_list: Optional[list]

    # Quality tracking
    retrieval_attempts: Optional[int]
    retrieval_quality_score: Optional[float]
    is_answer_sufficient: Optional[bool]
    final_answer: Optional[str]
    confidence_score: Optional[float]

    # Corpus metadata
    corpus_stats: Optional[dict]

    # Ground truth tracking
    ground_truth_doc_ids: Optional[list]


# ========== HELPER FUNCTIONS ==========

conversational_rewriter = ConversationalRewriter()
strategy_selector = StrategySelector()
cross_encoder = CrossEncoderReRanker()
adaptive_retriever = None


def extract_conversation_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Extract conversation turns from messages."""
    if not messages or len(messages) < 2:
        return []

    conversation = []
    i = 0

    while i < len(messages) - 1:
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i+1], AIMessage):
            conversation.append({
                "user": messages[i].content,
                "assistant": messages[i+1].content
            })
            i += 2
        else:
            i += 1

    return conversation


def _should_expand_query_llm(query: str) -> bool:
    """Use LLM to decide if query should be expanded."""
    spec = get_model_for_task("expansion_decision")
    model_kwargs = {}
    if spec.reasoning_effort:
        model_kwargs["reasoning_effort"] = spec.reasoning_effort
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        model_kwargs=model_kwargs
    )

    prompt = f"""Should this query be expanded into multiple variations?

Query: "{query}"

Expand if: ambiguous, could be phrased differently, needs synonyms
Skip if: clear, specific, well-formed, simple lookup

Return ONLY 'yes' or 'no'."""

    try:
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        return decision.startswith('yes')
    except Exception:
        return True  # Default to expand


class RetrievalQualityEval(TypedDict):
    """Binary retrieval quality assessment."""
    quality_score: float  # 0.0-1.0
    reasoning: str


class AnswerQualityEval(TypedDict):
    """Simple answer quality assessment."""
    is_sufficient: bool
    confidence_score: float
    reasoning: str


# ========== NODES ==========

def conversational_rewrite_node(state: IntermediateRAGState) -> dict:
    """Rewrite query using conversation history."""
    question = state.get("user_question", state.get("baseline_query", ""))
    messages = state.get("messages", [])
    conversation_history = extract_conversation_history(messages)

    rewritten_query, reasoning = conversational_rewriter.rewrite(
        question,
        conversation_history
    )

    if rewritten_query != question:
        print(f"\n{'='*60}")
        print(f"CONVERSATIONAL REWRITE")
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten_query}")
        print(f"{'='*60}\n")

    return {
        "baseline_query": rewritten_query,
        "user_question": question,
        "corpus_stats": get_corpus_stats(),
        "messages": [HumanMessage(content=question)],
    }


def decide_strategy_node(state: IntermediateRAGState) -> dict:
    """Select optimal retrieval strategy using LLM."""
    query = state.get("baseline_query", "")
    corpus_stats = state.get("corpus_stats", {})

    # Unpack all 3 return values from strategy selector
    strategy, confidence, reasoning = strategy_selector.select_strategy(query, corpus_stats)

    print(f"\n{'='*60}")
    print(f"STRATEGY SELECTION")
    print(f"Query: {query}")
    print(f"Selected: {strategy} (confidence: {confidence:.0%})")
    print(f"Reasoning: {reasoning}")
    print(f"{'='*60}\n")

    return {"retrieval_strategy": strategy}


def query_expansion_node(state: IntermediateRAGState) -> dict:
    """Conditionally expand query using LLM decision."""
    query = state.get("baseline_query", "")

    if _should_expand_query_llm(query):
        # expand_query doesn't accept num_variations parameter
        expansions = expand_query(query)
        print(f"\n{'='*60}")
        print(f"QUERY EXPANSION")
        print(f"Original: {query}")
        print(f"Expansions: {len(expansions)}")
        for i, exp in enumerate(expansions, 1):
            print(f"  {i}. {exp}")
        print(f"{'='*60}\n")
    else:
        expansions = [query]  # Include original query
        print(f"\n{'='*60}")
        print(f"QUERY EXPANSION SKIPPED")
        print(f"Reason: Query is clear and specific")
        print(f"{'='*60}\n")

    return {
        "active_query": query,
        "query_expansions": expansions,
    }


def retrieve_with_expansion_node(state: IntermediateRAGState) -> dict:
    """Retrieve using selected strategy with RRF fusion."""
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    query = state.get("active_query", state["baseline_query"])
    expansions = state.get("query_expansions", [])
    strategy = state.get("retrieval_strategy", "hybrid")
    attempts = state.get("retrieval_attempts", 0)

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    # Retrieve using selected strategy with RRF fusion
    if expansions and len(expansions) > 1:
        # RRF fusion implementation (inline, as in nodes.py)
        doc_ranks = {}
        doc_objects = {}

        for q in expansions:
            docs = adaptive_retriever.retrieve_without_reranking(q, strategy=strategy)

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
        # Pre-reranking pool: 15 docs for standard (k=4), scales for larger k
        pool_size = 15 if adaptive_retriever and adaptive_retriever.k_final <= 4 else min(20, len(sorted_doc_ids))
        top_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids[:pool_size]]

        print(f"\n{'='*60}")
        print(f"RRF MULTI-QUERY RETRIEVAL")
        print(f"Strategy: {strategy}")
        print(f"Query variants: {len(expansions)}")
        print(f"Total retrievals: {sum(len(ranks) for ranks in doc_ranks.values())}")
        print(f"Unique docs after RRF: {len(sorted_doc_ids)}")

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

        print(f"Sending top {pool_size} docs to reranking")
        print(f"{'='*60}\n")
    else:
        top_docs = adaptive_retriever.retrieve_without_reranking(query, strategy=strategy)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL")
    print(f"Strategy: {strategy}")
    print(f"Attempt: {attempts + 1}")
    print(f"Retrieved: {len(top_docs)} documents")
    print(f"{'='*60}\n")

    unique_docs = list({doc.page_content: doc for doc in top_docs}.values())

    return {
        "retrieved_docs": [doc.page_content for doc in unique_docs],
        "unique_docs_list": unique_docs,
        "retrieval_attempts": attempts + 1,
    }


def rerank_node(state: IntermediateRAGState) -> dict:
    """Two-stage reranking: CrossEncoder → LLM-as-judge."""
    global adaptive_retriever
    query = state.get("active_query", state["baseline_query"])
    docs = state.get("unique_docs_list", [])

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    # Stage 1: CrossEncoder (top-15)
    stage1_results = cross_encoder.rank(query, docs[:15])
    stage1_docs = [doc for doc, score in stage1_results]

    # Stage 2: LLM-as-judge (top-4)
    spec = get_model_for_task("llm_reranking")
    model_kwargs = {}
    if spec.reasoning_effort:
        model_kwargs["reasoning_effort"] = spec.reasoning_effort
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        model_kwargs=model_kwargs
    )

    # Simple LLM scoring
    scored_docs = []
    for doc in stage1_docs[:8]:  # Score top-8 from stage 1
        prompt = f"""Rate relevance of this document to the query (0.0-1.0):

Query: {query}

Document: {doc.page_content[:500]}...

Return only a score between 0.0 and 1.0."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            scored_docs.append((doc, score))
        except:
            scored_docs.append((doc, 0.5))

    # Sort by score and take top-k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    k_final = adaptive_retriever.k_final if adaptive_retriever else 4
    final_docs = [doc for doc, _ in scored_docs[:k_final]]

    print(f"\n{'='*60}")
    print(f"TWO-STAGE RERANKING (After RRF)")
    print(f"Input: {len(docs)} docs (from RRF)")

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

    print(f"\nStage 1 (CrossEncoder): {len(docs)} -> {len(stage1_docs)}")
    print(f"Stage 2 (LLM-as-judge): {len(stage1_docs)} -> {len(final_docs)}")
    print(f"Output: {len(final_docs)} docs after two-stage reranking")

    # Show final chunk IDs with LLM-as-judge scores
    print(f"\nFinal chunk IDs (after two-stage reranking):")
    for i, (doc, score) in enumerate(scored_docs[:k_final], 1):
        chunk_id = doc.metadata.get("id", "unknown")
        print(f"  {i}. {chunk_id} (score: {score:.4f})")

    # Track ground truth in final results
    if ground_truth_doc_ids:
        final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in final_docs]
        found_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in final_chunk_ids]
        missing_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in final_chunk_ids]
        print(f"\nExpected chunks in final results:")
        print(f"Found: {found_in_final if found_in_final else '[]'} | Missing: {missing_in_final if missing_in_final else '[]'}")

    print(f"{'='*60}\n")

    return {"unique_docs_list": final_docs}


def grade_documents_node(state: IntermediateRAGState) -> dict:
    """Binary quality assessment of retrieved documents."""
    query = state.get("active_query", state["baseline_query"])
    docs = state.get("unique_docs_list", [])

    spec = get_model_for_task("retrieval_quality_eval")
    model_kwargs = {}
    if spec.reasoning_effort:
        model_kwargs["reasoning_effort"] = spec.reasoning_effort
    grader = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        model_kwargs=model_kwargs
    ).with_structured_output(RetrievalQualityEval)

    combined_docs = "\n\n".join([doc.page_content[:300] for doc in docs])

    prompt = f"""Assess retrieval quality for this query.

Query: {query}

Retrieved Documents:
{combined_docs}

Provide:
- quality_score: 0.0 (poor) to 1.0 (excellent)
- reasoning: Why this score?"""

    try:
        result = grader.invoke(prompt)
        quality_score = result["quality_score"]
        reasoning = result["reasoning"]
    except:
        quality_score = 0.7
        reasoning = "Default score (LLM grading failed)"

    print(f"\n{'='*60}")
    print(f"RETRIEVAL QUALITY")
    print(f"Score: {quality_score:.0%}")
    print(f"Reasoning: {reasoning}")
    print(f"{'='*60}\n")

    return {"retrieval_quality_score": quality_score}


def answer_generation_node(state: IntermediateRAGState) -> dict:
    """Generate answer and assess quality."""
    query = state.get("active_query", state["baseline_query"])
    docs = state.get("unique_docs_list", [])
    attempts = state.get("retrieval_attempts", 0)

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

    answer_prompt = f"""Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Provide a comprehensive answer based strictly on the context."""

    answer = llm.invoke(answer_prompt).content

    # Assess answer quality
    evaluator = llm.with_structured_output(AnswerQualityEval)

    eval_prompt = f"""Assess if this answer is sufficient.

Question: {query}
Answer: {answer}
Context: {context[:500]}...

Provide:
- is_sufficient: true if answer fully addresses question
- confidence_score: 0.0-1.0
- reasoning: Why?"""

    try:
        eval_result = evaluator.invoke(eval_prompt)
        is_sufficient = eval_result["is_sufficient"]
        confidence = eval_result["confidence_score"]
    except:
        is_sufficient = True  # Default to sufficient if eval fails
        confidence = 0.7

    print(f"\n{'='*60}")
    print(f"ANSWER GENERATION")
    print(f"Answer length: {len(answer)} chars")
    print(f"Sufficient: {is_sufficient}")
    print(f"Confidence: {confidence:.0%}")
    print(f"{'='*60}\n")

    return {
        "final_answer": answer,
        "is_answer_sufficient": is_sufficient,
        "confidence_score": confidence,
        "messages": [AIMessage(content=answer)],
    }


def rewrite_query_node(state: IntermediateRAGState) -> dict:
    """Rewrite query with generic feedback."""
    query = state.get("active_query", state["baseline_query"])
    quality_score = state.get("retrieval_quality_score", 0.7)

    # Generic feedback
    feedback = f"Previous retrieval quality was {quality_score:.0%}. Try rephrasing to improve results."

    # Use correct parameter name: retrieval_context
    rewritten = rewrite_query(query, retrieval_context=feedback)

    print(f"\n{'='*60}")
    print(f"QUERY REWRITE")
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten}")
    print(f"{'='*60}\n")

    return {
        "active_query": rewritten,
        "query_expansions": [],  # Clear stale expansions after rewrite
    }


# ========== ROUTING FUNCTIONS ==========

def route_after_retrieval(state: IntermediateRAGState) -> Literal["answer_generation", "rewrite_query"]:
    """Route based on retrieval quality."""
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)

    if quality >= 0.6 or attempts >= 2:
        return "answer_generation"
    else:
        return "rewrite_query"


def route_after_generation(state: IntermediateRAGState) -> Literal["rewrite_query", "END"]:
    """Route based on answer quality."""
    is_sufficient = state.get("is_answer_sufficient", True)
    attempts = state.get("retrieval_attempts", 0)

    if is_sufficient or attempts >= 2:  # Max 1 rewrite (total 2 attempts)
        return END
    else:
        return "rewrite_query"


# ========== GRAPH BUILDER ==========

def build_intermediate_rag_graph():
    """Build intermediate RAG graph with quality gates."""
    builder = StateGraph(IntermediateRAGState)

    # Add nodes
    builder.add_node("conversational_rewrite", conversational_rewrite_node)
    builder.add_node("query_expansion", query_expansion_node)
    builder.add_node("decide_strategy", decide_strategy_node)
    builder.add_node("retrieve_with_expansion", retrieve_with_expansion_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("rewrite_query", rewrite_query_node)

    # Build flow
    builder.add_edge(START, "conversational_rewrite")
    builder.add_edge("conversational_rewrite", "query_expansion")
    builder.add_edge("query_expansion", "decide_strategy")
    builder.add_edge("decide_strategy", "retrieve_with_expansion")
    builder.add_edge("retrieve_with_expansion", "rerank")
    builder.add_edge("rerank", "grade_documents")

    # Conditional routing
    builder.add_conditional_edges(
        "grade_documents",
        route_after_retrieval,
        {
            "answer_generation": "answer_generation",
            "rewrite_query": "rewrite_query",
        }
    )

    builder.add_edge("rewrite_query", "query_expansion")

    builder.add_conditional_edges(
        "answer_generation",
        route_after_generation,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        }
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


intermediate_rag_graph = build_intermediate_rag_graph()
