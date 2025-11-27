"""
Multi-Agent RAG Graph (20 Features) - Orchestrator-Worker Pattern.

For complex queries requiring multi-faceted retrieval.
Decomposes query into sub-queries, parallel worker retrieval, RRF fusion.

Features (20 = +3 over Advanced):

Inherited from Advanced (17):
1. Semantic vector search
2. Query expansion (multi-variant)
3. Hybrid retrieval (semantic + BM25)
4. RRF fusion
5. CrossEncoder reranking
6. Conversational query rewriting
7. LLM-based strategy selection
8. Two-stage reranking (CrossEncoder -> LLM-as-judge)
9. Retrieval quality gates (8 issue types)
10. Answer quality evaluation (8 issue types)
11. Adaptive thresholds (65%/50%)
12. Query rewriting loop (issue-specific feedback)
13. Early strategy switching (off_topic/wrong_domain)
14. Generation retry loop (adaptive temperature)
15. NLI-based hallucination detection
16. Refusal detection
17. Conversation context preservation

Multi-Agent Specific (+3):
18. Complexity classification (simple vs complex routing)
19. Query decomposition (2-4 sub-queries)
20. Parallel worker retrieval with RRF merge + LLM coverage selection

Graph Structure: 7 nodes, orchestrator-worker pattern
- conversational_rewrite_node
- classify_complexity_node (orchestrator decision)
- decompose_query_node (orchestrator)
- retrieval_worker (parallel workers via Send API)
- merge_results_node (synthesizer with RRF + LLM coverage)
- answer_generation_node
- evaluate_answer_node

Routing Functions:
- assign_workers: Fan-out to parallel retrieval workers
- route_after_evaluation: Generation retry or end

Pattern: Orchestrator-Worker (LangGraph docs)
https://docs.langchain.com/oss/python/langgraph/workflows-agents

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

from typing import TypedDict, Optional, Literal, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from advanced_agentic_rag_langgraph.retrieval import expand_query, rewrite_query
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
from advanced_agentic_rag_langgraph.retrieval.query_optimization import optimize_query_for_strategy
from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter
from advanced_agentic_rag_langgraph.prompts import get_prompt
from advanced_agentic_rag_langgraph.prompts.answer_generation import get_answer_generation_prompts
from advanced_agentic_rag_langgraph.validation import NLIHallucinationDetector
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_ndcg


# ========== GLOBALS ==========

adaptive_retriever = None
conversational_rewriter = ConversationalRewriter()
strategy_selector = StrategySelector()
nli_detector = NLIHallucinationDetector()


# ========== STATE SCHEMAS ==========

class MultiAgentRAGState(TypedDict):
    """State for multi-agent RAG variant."""
    # Input
    user_question: str
    baseline_query: Optional[str]
    corpus_stats: Optional[dict]

    # Complexity routing
    is_complex_query: Optional[bool]
    complexity_reasoning: Optional[str]
    sub_queries: Optional[list[str]]

    # Worker results (accumulated via operator.add)
    sub_agent_results: Annotated[list[dict], operator.add]

    # Merged results
    retrieved_docs: Optional[list]
    unique_docs_list: Optional[list]
    retrieval_quality_score: Optional[float]
    multi_agent_metrics: Optional[dict]

    # Answer generation
    final_answer: Optional[str]
    confidence_score: Optional[float]
    generation_attempts: Optional[int]
    retry_feedback: Optional[str]

    # Answer evaluation
    is_refusal: Optional[bool]
    is_answer_sufficient: Optional[bool]
    groundedness_score: Optional[float]
    has_hallucination: Optional[bool]
    unsupported_claims: Optional[list[str]]
    answer_quality_reasoning: Optional[str]
    answer_quality_issues: Optional[list[str]]

    # Evaluation support
    ground_truth_doc_ids: Optional[list]
    relevance_grades: Optional[dict]

    # Conversation history (LangGraph best practice)
    messages: Annotated[list[BaseMessage], operator.add]


class WorkerState(TypedDict):
    """State passed to each retrieval worker via Send."""
    sub_query: str
    corpus_stats: Optional[dict]
    worker_index: int
    # Reducer for accumulation back to parent
    sub_agent_results: Annotated[list[dict], operator.add]


class RetrievalSubgraphState(TypedDict):
    """Isolated state for retrieval subgraph."""
    # Input
    sub_query: str
    corpus_stats: Optional[dict]

    # Strategy selection
    retrieval_strategy: Optional[Literal["semantic", "keyword", "hybrid"]]

    # Query lifecycle
    active_query: Optional[str]
    retrieval_query: Optional[str]
    query_expansions: Optional[list[str]]

    # Retrieval
    retrieved_docs: Optional[list]  # Replaced each attempt (not accumulated)
    retrieval_attempts: int

    # Quality
    retrieval_quality_score: Optional[float]
    retrieval_quality_issues: Optional[list[str]]
    retrieval_improvement_suggestion: Optional[str]

    # Output
    final_docs: Optional[list]


# ========== STRUCTURED OUTPUT SCHEMAS ==========

class ComplexityDecision(TypedDict):
    """Structured output for complexity classification."""
    is_complex: bool
    reasoning: str


class QueryDecomposition(TypedDict):
    """Structured output for query decomposition."""
    sub_queries: list[str]
    reasoning: str


class RetrievalQualityEvaluation(TypedDict):
    """Structured output for retrieval quality assessment."""
    quality_score: float
    reasoning: str
    issues: list[str]
    improvement_suggestion: str


class AnswerQualityEvaluation(TypedDict):
    """Structured output for answer quality assessment."""
    is_relevant: bool
    is_complete: bool
    is_accurate: bool
    confidence_score: float
    reasoning: str
    issues: list[str]


class RefusalCheck(TypedDict):
    """Structured output for refusal detection."""
    refused: bool
    reasoning: str


# ========== HELPER FUNCTIONS ==========

def _extract_conversation_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Extract conversation history from messages list."""
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


# ========== CONVERSATIONAL PREPROCESSING ==========

def conversational_rewrite_node(state: MultiAgentRAGState) -> dict:
    """Rewrite query using conversation history to make it self-contained."""
    question = state.get("user_question", "")
    messages = state.get("messages", [])
    conversation_history = _extract_conversation_history(messages)

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
        "corpus_stats": get_corpus_stats(),
        "messages": [HumanMessage(content=question)],
        # Reset state for new question
        "generation_attempts": 0,
        "retry_feedback": None,
        "is_refusal": None,
        "is_answer_sufficient": None,
        "final_answer": None,
        "confidence_score": None,
        "sub_agent_results": [],  # Clear previous results
    }


# ========== COMPLEXITY CLASSIFICATION ==========

def classify_complexity_node(state: MultiAgentRAGState) -> dict:
    """
    Classify query complexity to determine routing.

    Pure node - stores result in state for downstream routing.
    Complex queries get decomposed; simple queries could use single-hop
    (but this variant always uses multi-agent for demonstration).
    """
    question = state.get("baseline_query", state.get("user_question", ""))

    spec = get_model_for_task("complexity_classification")
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
    )
    structured_llm = llm.with_structured_output(ComplexityDecision)

    prompt = f"""Classify if this query requires multi-faceted retrieval (complex) or single-aspect lookup (simple).

Query: "{question}"

COMPLEX INDICATORS (decomposition beneficial):
- Comparative questions ("Compare X and Y", "differences between")
- Multi-aspect queries ("What are the components AND how do they interact")
- Cross-domain synthesis ("How does X in domain A relate to Y in domain B")
- Questions requiring information from multiple document sections
- Research questions with multiple sub-questions implied

SIMPLE INDICATORS (single retrieval sufficient):
- Single-aspect factual lookup ("What is X?", "Define Y")
- Procedural questions with linear steps ("How do I do X?")
- Specific detail requests ("When was X published?")
- Questions targeting a single concept or definition

Return is_complex=True if decomposition would improve retrieval coverage."""

    try:
        result = structured_llm.invoke(prompt)
        is_complex = result["is_complex"]
        reasoning = result["reasoning"]
    except Exception as e:
        print(f"Warning: Complexity classification failed: {e}. Defaulting to complex.")
        is_complex = True
        reasoning = f"Classification failed: {e}"

    print(f"\n{'='*60}")
    print(f"COMPLEXITY CLASSIFICATION")
    print(f"Query: {question}")
    print(f"Classification: {'COMPLEX' if is_complex else 'SIMPLE'}")
    print(f"Reasoning: {reasoning}")
    print(f"{'='*60}\n")

    return {
        "is_complex_query": is_complex,
        "complexity_reasoning": reasoning,
        "messages": [AIMessage(content=f"Complexity: {'complex' if is_complex else 'simple'}")],
    }


# ========== QUERY DECOMPOSITION (ORCHESTRATOR) ==========

def decompose_query_node(state: MultiAgentRAGState) -> dict:
    """
    Decompose complex query into 2-4 focused sub-queries.

    ORCHESTRATOR in the orchestrator-worker pattern.
    Each sub-query targets a distinct aspect for parallel retrieval.
    """
    question = state.get("baseline_query", state.get("user_question", ""))

    spec = get_model_for_task("query_decomposition")
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
    )
    structured_llm = llm.with_structured_output(QueryDecomposition)

    prompt = f"""Decompose this query into 2-4 focused sub-queries for parallel retrieval.

Query: "{question}"

DECOMPOSITION GUIDELINES:
1. Each sub-query targets a DISTINCT aspect of the original question
2. Sub-queries are SELF-CONTAINED (can be understood without the original)
3. Sub-queries are PARALLEL (not dependent on each other's results)
4. Preserve technical terms and domain-specific vocabulary exactly
5. Aim for 2-4 sub-queries (more is not always better)

EXAMPLES:
- "What are the benefits and limitations of renewable energy compared to fossil fuels?" ->
  ["What are the main benefits of renewable energy sources?",
   "What are the limitations or challenges of renewable energy?",
   "What are the advantages and disadvantages of fossil fuels?"]

- "How does machine learning improve medical diagnosis accuracy?" ->
  ["What machine learning techniques are used in medical diagnosis?",
   "How is diagnostic accuracy measured in healthcare?",
   "What improvements in accuracy have been achieved with ML-based diagnosis?"]

Return sub-queries as a list of strings."""

    try:
        result = structured_llm.invoke(prompt)
        sub_queries = result["sub_queries"]
        reasoning = result["reasoning"]
    except Exception as e:
        print(f"Warning: Query decomposition failed: {e}. Using original query.")
        sub_queries = [question]
        reasoning = f"Decomposition failed: {e}"

    print(f"\n{'='*60}")
    print(f"QUERY DECOMPOSITION (ORCHESTRATOR)")
    print(f"Original: {question}")
    print(f"Sub-queries ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    print(f"Reasoning: {reasoning}")
    print(f"{'='*60}\n")

    return {
        "sub_queries": sub_queries,
        "messages": [AIMessage(content=f"Decomposed into {len(sub_queries)} sub-queries")],
    }


# ========== WORKER ASSIGNMENT (Send API) ==========

def assign_workers(state: MultiAgentRAGState) -> list[Send]:
    """
    Assign a retrieval worker to each sub-query.

    Returns list[Send] for parallel execution via LangGraph Send API.
    Each worker receives its sub-query and corpus stats.
    """
    sub_queries = state.get("sub_queries", [])
    corpus_stats = state.get("corpus_stats", {})

    print(f"\n{'='*60}")
    print(f"ASSIGN WORKERS (Send API)")
    print(f"Spawning {len(sub_queries)} parallel retrieval workers")
    print(f"{'='*60}\n")

    return [
        Send("retrieval_worker", {
            "sub_query": sq,
            "corpus_stats": corpus_stats,
            "worker_index": i,
        })
        for i, sq in enumerate(sub_queries)
    ]


# ========== RETRIEVAL WORKER SUBGRAPH ==========

def _build_retrieval_subgraph():
    """
    Build isolated retrieval subgraph with self-correction loop.

    Flow: decide_strategy -> query_expansion -> retrieve -> [quality check]
                                                               |
                                            [>= 0.6 OR attempts >= 2] -> finalize
                                            [< 0.6 AND attempts < 2] -> rewrite -> query_expansion
    """
    builder = StateGraph(RetrievalSubgraphState)

    # Nodes
    builder.add_node("decide_strategy", _subgraph_decide_strategy_node)
    builder.add_node("query_expansion", _subgraph_query_expansion_node)
    builder.add_node("retrieve", _subgraph_retrieve_node)
    builder.add_node("rewrite_and_refine", _subgraph_rewrite_node)
    builder.add_node("finalize", _subgraph_finalize_node)

    # Flow
    builder.add_edge(START, "decide_strategy")
    builder.add_edge("decide_strategy", "query_expansion")
    builder.add_edge("query_expansion", "retrieve")

    builder.add_conditional_edges(
        "retrieve",
        _subgraph_route_after_retrieval,
        {
            "finalize": "finalize",
            "rewrite_and_refine": "rewrite_and_refine",
        }
    )

    builder.add_edge("rewrite_and_refine", "query_expansion")
    builder.add_edge("finalize", END)

    return builder.compile()


def _subgraph_route_after_retrieval(state: RetrievalSubgraphState) -> Literal["finalize", "rewrite_and_refine"]:
    """Pure router for subgraph retrieval quality check."""
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)

    # Max 2 attempts (aligned with main graph)
    if quality >= 0.6 or attempts >= 2:
        return "finalize"
    return "rewrite_and_refine"


def _subgraph_decide_strategy_node(state: RetrievalSubgraphState) -> dict:
    """Decide retrieval strategy for sub-query."""
    query = state["sub_query"]
    corpus_stats = state.get("corpus_stats", {})

    strategy, confidence, reasoning = strategy_selector.select_strategy(
        query,
        corpus_stats
    )

    print(f"  [Worker] Strategy: {strategy} ({confidence:.0%})")

    return {
        "retrieval_strategy": strategy,
        "active_query": query,
    }


def _subgraph_query_expansion_node(state: RetrievalSubgraphState) -> dict:
    """Optimize and expand query for retrieval."""
    query = state.get("active_query", state["sub_query"])
    strategy = state.get("retrieval_strategy", "hybrid")

    optimized_query = optimize_query_for_strategy(
        query=query,
        strategy=strategy,
    )

    expansions = expand_query(optimized_query)

    return {
        "retrieval_query": optimized_query,
        "query_expansions": expansions,
    }


def _subgraph_retrieve_node(state: RetrievalSubgraphState) -> dict:
    """Execute RRF retrieval with two-stage reranking."""
    global adaptive_retriever

    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    strategy = state.get("retrieval_strategy", "hybrid")
    query = state.get("active_query", state["sub_query"])

    # RRF across query expansions
    doc_ranks = {}
    doc_objects = {}

    for expansion in state.get("query_expansions", [query]):
        docs = adaptive_retriever.retrieve_without_reranking(expansion, strategy=strategy)

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
        rrf_scores[doc_id] = sum(1.0 / (rank + k) for rank in ranks)

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    unique_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]

    # Two-stage reranking
    reranking_input = unique_docs[:40]
    ranked_results = adaptive_retriever.reranker.rank(query, reranking_input)
    final_docs = [doc for doc, score in ranked_results]

    # Quality evaluation
    docs_text = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in final_docs
    ])

    spec = get_model_for_task("retrieval_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
    )
    structured_llm = quality_llm.with_structured_output(RetrievalQualityEvaluation)

    quality_prompt = get_prompt("retrieval_quality_eval", query=query, docs_text=docs_text)

    try:
        evaluation = structured_llm.invoke(quality_prompt)
        quality_score = evaluation["quality_score"] / 100
        quality_issues = evaluation["issues"]
        improvement_suggestion = evaluation.get("improvement_suggestion", "")
    except Exception as e:
        print(f"  [Worker] Quality evaluation failed: {e}")
        quality_score = 0.5
        quality_issues = []
        improvement_suggestion = ""

    attempts = state.get("retrieval_attempts", 0) + 1
    print(f"  [Worker] Retrieved {len(final_docs)} docs, quality: {quality_score:.0%}, attempt: {attempts}/2")

    return {
        "retrieved_docs": final_docs,
        "retrieval_quality_score": quality_score,
        "retrieval_quality_issues": quality_issues,
        "retrieval_improvement_suggestion": improvement_suggestion,
        "retrieval_attempts": attempts,
    }


def _subgraph_rewrite_node(state: RetrievalSubgraphState) -> dict:
    """Rewrite query based on quality issues."""
    query = state.get("active_query", state["sub_query"])
    suggestion = state.get("retrieval_improvement_suggestion", "")
    issues = state.get("retrieval_quality_issues", [])

    retrieval_context = f"""Previous retrieval quality: {state.get('retrieval_quality_score', 0):.0%}
Improvement needed: {suggestion}
Issues: {', '.join(issues) if issues else 'None'}"""

    rewritten = rewrite_query(query, retrieval_context=retrieval_context)
    print(f"  [Worker] Rewritten: {query[:50]}... -> {rewritten[:50]}...")

    return {
        "active_query": rewritten,
        "query_expansions": [],
        "retrieval_query": None,
    }


def _subgraph_finalize_node(state: RetrievalSubgraphState) -> dict:
    """Package final documents for return to parent."""
    return {
        "final_docs": state.get("retrieved_docs", []),
    }


# Build subgraph once
retrieval_subgraph = _build_retrieval_subgraph()


# ========== RETRIEVAL WORKER NODE ==========

def retrieval_worker(state: WorkerState) -> dict:
    """
    Worker that executes full retrieval pipeline for a sub-query.

    Invokes retrieval_subgraph with strategy selection, retrieval, and retry loop.
    Returns results for accumulation via operator.add.
    """
    sub_query = state["sub_query"]
    corpus_stats = state.get("corpus_stats", {})
    worker_index = state.get("worker_index", 0)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL WORKER {worker_index}")
    print(f"Sub-query: {sub_query}")
    print(f"{'='*60}")

    try:
        # Invoke the retrieval subgraph
        result = retrieval_subgraph.invoke({
            "sub_query": sub_query,
            "corpus_stats": corpus_stats,
            "retrieval_attempts": 0,
        })

        docs = result.get("final_docs", [])
        quality = result.get("retrieval_quality_score", 0)

        print(f"Worker {worker_index} complete: {len(docs)} docs, quality: {quality:.0%}")

        return {
            "sub_agent_results": [{
                "sub_query": sub_query,
                "docs": docs,
                "quality_score": quality,
                "worker_index": worker_index,
            }],
        }

    except Exception as e:
        print(f"Worker {worker_index} failed: {e}")
        return {
            "sub_agent_results": [{
                "sub_query": sub_query,
                "docs": [],
                "quality_score": 0.0,
                "worker_index": worker_index,
                "error": str(e),
            }],
        }


# ========== RESULT MERGING (SYNTHESIZER) ==========

def merge_results_node(state: MultiAgentRAGState) -> dict:
    """
    Merge results from all parallel sub-agents using RRF + LLM coverage selection.

    SYNTHESIZER in the orchestrator-worker pattern.
    Two-stage merge:
    1. RRF fusion to get top-12 candidates (fast pre-filter)
    2. LLM coverage-aware selection to get top-6 (semantic understanding)

    Research: SetR/PureCover papers on coverage-based selection for multi-hop QA.
    """
    sub_agent_results = state.get("sub_agent_results", [])
    original_question = state.get("baseline_query", state.get("user_question", ""))
    sub_queries = state.get("sub_queries", [])

    print(f"\n{'='*60}")
    print(f"MERGE RESULTS (SYNTHESIZER)")
    print(f"Merging results from {len(sub_agent_results)} workers")

    if not sub_agent_results:
        print("No results to merge!")
        print(f"{'='*60}\n")
        return {
            "retrieved_docs": [],
            "unique_docs_list": [],
            "retrieval_quality_score": 0.0,
            "multi_agent_metrics": {"workers": 0, "total_docs": 0},
        }

    # ========== STAGE 1: Cross-agent RRF fusion ==========
    doc_ranks = {}
    doc_objects = {}
    doc_agent_count = {}  # Track which docs appear in multiple agents

    for result in sub_agent_results:
        worker_idx = result.get("worker_index", 0)
        docs = result.get("docs", [])

        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])

            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
                doc_objects[doc_id] = doc
                doc_agent_count[doc_id] = set()

            doc_ranks[doc_id].append(rank)
            doc_agent_count[doc_id].add(worker_idx)

    # RRF scoring (no multi-agent boost - let LLM coverage selection handle diversity)
    k = 60
    rrf_scores = {}
    for doc_id, ranks in doc_ranks.items():
        rrf_scores[doc_id] = sum(1.0 / (rank + k) for rank in ranks)

    # Get top-12 candidates for LLM selection (or all if fewer)
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    candidate_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids[:12]]
    fallback_doc_ids = [f"doc_{i}" for i in range(min(6, len(candidate_docs)))]

    print(f"Stage 1 (RRF): {len(doc_objects)} unique docs -> {len(candidate_docs)} candidates")

    # Ground truth tracking after RRF stage
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])
    if ground_truth_doc_ids:
        candidate_chunk_ids = [doc.metadata.get("id", "unknown") for doc in candidate_docs]
        found_in_rrf = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in candidate_chunk_ids]
        missing_in_rrf = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in candidate_chunk_ids]
        print(f"\nExpected chunks in RRF candidates:")
        print(f"Found: {found_in_rrf if found_in_rrf else '[]'} | Missing: {missing_in_rrf if missing_in_rrf else '[]'}")

    # ========== STAGE 2: LLM coverage-aware selection ==========
    from advanced_agentic_rag_langgraph.retrieval.multi_agent_merge_reranker import MultiAgentMergeReRanker

    coverage_reranker = MultiAgentMergeReRanker(top_k=6)
    top_docs = coverage_reranker.select_for_coverage(
        original_question=original_question,
        sub_queries=sub_queries or [],
        candidate_docs=candidate_docs,
        fallback_doc_ids=fallback_doc_ids,
    )

    print(f"Stage 2 (LLM Coverage): {len(candidate_docs)} candidates -> {len(top_docs)} selected")

    # Ground truth tracking after LLM coverage selection
    if ground_truth_doc_ids:
        final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in top_docs]
        found_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in final_chunk_ids]
        missing_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in final_chunk_ids]
        print(f"\nExpected chunks in final selection:")
        print(f"Found: {found_in_final if found_in_final else '[]'} | Missing: {missing_in_final if missing_in_final else '[]'}")

    # Calculate average quality
    quality_scores = [r.get("quality_score", 0) for r in sub_agent_results if r.get("docs")]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Format for answer generation
    docs_text = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in top_docs
    ])

    multi_agent_metrics = {
        "workers": len(sub_agent_results),
        "total_unique_docs": len(doc_objects),
        "multi_agent_docs": sum(1 for count in doc_agent_count.values() if len(count) > 1),
        "top_k_selected": len(top_docs),
        "avg_quality": avg_quality,
        "merge_method": "rrf_plus_llm_coverage",
    }

    print(f"Total unique docs: {len(doc_objects)}")
    print(f"Multi-agent docs (in 2+ workers): {multi_agent_metrics['multi_agent_docs']}")
    print(f"Top-{len(top_docs)} selected for generation")
    print(f"Average quality: {avg_quality:.0%}")
    print(f"{'='*60}\n")

    return {
        "retrieved_docs": [docs_text],
        "unique_docs_list": top_docs,
        "retrieval_quality_score": avg_quality,
        "multi_agent_metrics": multi_agent_metrics,
        "messages": [AIMessage(content=f"Merged {len(doc_objects)} docs from {len(sub_agent_results)} workers")],
    }


# ========== ANSWER GENERATION ==========

def answer_generation_node(state: MultiAgentRAGState) -> dict:
    """Generate answer from merged multi-agent context."""
    question = state.get("baseline_query", state.get("user_question", ""))
    context = state["retrieved_docs"][-1] if state.get("retrieved_docs") else "No context"
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_attempts = state.get("generation_attempts", 0) + 1
    retry_feedback = state.get("retry_feedback", "")

    print(f"\n{'='*60}")
    print(f"ANSWER GENERATION")
    print(f"Question: {question}")
    print(f"Context size: {len(context)} chars")
    print(f"Retrieval quality: {retrieval_quality:.0%}")
    print(f"Generation attempt: {generation_attempts}/3")
    print(f"{'='*60}\n")

    if not context or context == "No context":
        return {
            "final_answer": "I could not retrieve relevant documents to answer your question.",
            "generation_attempts": generation_attempts,
            "messages": [AIMessage(content="Empty retrieval - no answer generated")],
        }

    # Quality instruction
    if generation_attempts > 1 and retry_feedback:
        quality_instruction = f"""RETRY GENERATION (Attempt {generation_attempts}/3)

Previous attempt had issues:
{retry_feedback}

Generate improved answer addressing ALL issues above."""
    elif retrieval_quality >= 0.8:
        quality_instruction = f"High Confidence Retrieval (Score: {retrieval_quality:.0%}). Answer directly."
    elif retrieval_quality >= 0.6:
        quality_instruction = f"Medium Confidence Retrieval (Score: {retrieval_quality:.0%}). Acknowledge gaps."
    else:
        quality_instruction = f"Low Confidence Retrieval (Score: {retrieval_quality:.0%}). Only answer what's supported."

    spec = get_model_for_task("answer_generation")
    is_gpt5 = spec.name.lower().startswith("gpt-5")

    system_prompt, user_message = get_answer_generation_prompts(
        hallucination_feedback="",
        quality_instruction=quality_instruction,
        formatted_context=context,
        question=question,
        is_gpt5=is_gpt5,
        is_retry_after_hallucination=False,
        unsupported_claims=None,
    )

    # Adaptive temperature
    attempt_temperatures = {1: 0.3, 2: 0.7, 3: 0.5}
    temperature = attempt_temperatures.get(generation_attempts, 0.7)

    llm = ChatOpenAI(
        model=spec.name,
        temperature=temperature,
    )
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])

    return {
        "final_answer": response.content,
        "generation_attempts": generation_attempts,
        "messages": [response],
    }


# ========== ANSWER EVALUATION ==========

def _get_quality_fix_guidance(issues: list[str]) -> str:
    """Generate fix guidance based on quality issues."""
    guidance_map = {
        "incomplete_synthesis": "Provide more comprehensive synthesis",
        "lacks_specificity": "Include specific details",
        "wrong_focus": "Re-read question and address primary intent",
        "partial_answer": "Ensure all question parts are answered",
        "missing_details": "Add more depth where context supports",
    }
    return "; ".join([guidance_map.get(issue, issue) for issue in issues])


def evaluate_answer_node(state: MultiAgentRAGState) -> dict:
    """Combined refusal + groundedness + quality evaluation."""
    answer = state.get("final_answer", "")
    context = state.get("retrieved_docs", [""])[-1]
    question = state.get("baseline_query", state.get("user_question", ""))
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_attempts = state.get("generation_attempts", 0)

    print(f"\n{'='*60}")
    print(f"ANSWER EVALUATION")
    print(f"Generation attempt: {generation_attempts}")

    # 1. Refusal detection
    refusal_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    refusal_checker = refusal_llm.with_structured_output(RefusalCheck)

    try:
        refusal_result = refusal_checker.invoke([{
            "role": "system",
            "content": "Detect if the answer is a FULL REFUSAL (no substantive information provided)."
        }, {
            "role": "user",
            "content": f"Question: {question}\n\nAnswer: {answer}\n\nIs this a full refusal?"
        }])
        is_refusal = refusal_result["refused"]
    except Exception as e:
        is_refusal = False

    if is_refusal:
        print(f"Refusal detected - terminating")
        print(f"{'='*60}\n")
        return {
            "is_refusal": True,
            "is_answer_sufficient": False,
            "groundedness_score": 1.0,
            "has_hallucination": False,
            "unsupported_claims": [],
            "confidence_score": 0.0,
            "retry_feedback": "",
            "messages": [AIMessage(content="Refusal detected")],
        }

    # 2. Groundedness check
    groundedness_result = nli_detector.verify_groundedness(answer, context)
    groundedness_score = groundedness_result.get("groundedness_score", 1.0)
    has_hallucination = groundedness_score < 0.8
    unsupported_claims = groundedness_result.get("unsupported_claims", [])
    print(f"Groundedness: {groundedness_score:.0%}")

    # 3. Quality check
    quality_threshold = 0.5 if retrieval_quality < 0.6 else 0.65

    spec = get_model_for_task("answer_quality_eval")
    quality_llm = ChatOpenAI(model=spec.name, temperature=spec.temperature)
    structured_llm = quality_llm.with_structured_output(AnswerQualityEvaluation)

    evaluation_prompt = get_prompt(
        "answer_quality_eval",
        question=question,
        answer=answer,
        retrieval_quality=f"{retrieval_quality:.0%}",
        retrieval_issues="None",
        quality_threshold_pct=quality_threshold*100,
        quality_threshold_low_pct=(quality_threshold-0.15)*100,
        quality_threshold_minus_1_pct=quality_threshold*100-1,
        quality_threshold_low_minus_1_pct=(quality_threshold-0.15)*100-1
    )

    try:
        evaluation = structured_llm.invoke(evaluation_prompt)
        confidence = evaluation["confidence_score"] / 100
        quality_issues = evaluation["issues"]
    except Exception as e:
        confidence = 0.5
        quality_issues = []

    is_quality_sufficient = confidence >= quality_threshold
    print(f"Quality: {confidence:.0%}")

    # 4. Combined decision
    has_issues = has_hallucination or not is_quality_sufficient

    retry_feedback_parts = []
    if has_hallucination:
        retry_feedback_parts.append(
            f"HALLUCINATION DETECTED ({groundedness_score:.0%} grounded):\n"
            f"Unsupported claims: {', '.join(unsupported_claims)}"
        )
    if not is_quality_sufficient:
        retry_feedback_parts.append(
            f"QUALITY ISSUES:\n"
            f"Problems: {', '.join(quality_issues)}\n"
            f"Fix: {_get_quality_fix_guidance(quality_issues)}"
        )

    print(f"Decision: {'RETRY' if has_issues else 'SUFFICIENT'}")
    print(f"{'='*60}\n")

    return {
        "is_answer_sufficient": not has_issues,
        "is_refusal": False,
        "groundedness_score": groundedness_score,
        "has_hallucination": has_hallucination,
        "unsupported_claims": unsupported_claims,
        "confidence_score": confidence,
        "answer_quality_issues": quality_issues,
        "retry_feedback": "\n\n".join(retry_feedback_parts) if retry_feedback_parts else "",
        "messages": [AIMessage(content=f"Evaluation: {groundedness_score:.0%} grounded, {confidence:.0%} quality")],
    }


# ========== ROUTING FUNCTIONS ==========

def route_after_evaluation(state: MultiAgentRAGState) -> Literal["answer_generation", "END"]:
    """Route based on evaluation: retry generation or end."""
    if state.get("is_refusal", False):
        return END

    if state.get("is_answer_sufficient"):
        return END

    generation_attempts = state.get("generation_attempts", 0)
    if generation_attempts < 3:
        print(f"\nRouting: answer_generation (attempt {generation_attempts + 1}/3)")
        return "answer_generation"

    return END


# ========== GRAPH BUILDER ==========

def build_multi_agent_rag_graph():
    """Build multi-agent RAG graph with orchestrator-worker pattern."""
    builder = StateGraph(MultiAgentRAGState)

    # ========== NODES ==========
    builder.add_node("conversational_rewrite", conversational_rewrite_node)
    builder.add_node("classify_complexity", classify_complexity_node)
    builder.add_node("decompose_query", decompose_query_node)
    builder.add_node("retrieval_worker", retrieval_worker)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)

    # ========== FLOW ==========
    builder.add_edge(START, "conversational_rewrite")
    builder.add_edge("conversational_rewrite", "classify_complexity")
    builder.add_edge("classify_complexity", "decompose_query")

    # Orchestrator -> Workers (parallel via Send)
    builder.add_conditional_edges(
        "decompose_query",
        assign_workers,
        ["retrieval_worker"]
    )

    # Workers -> Synthesizer
    builder.add_edge("retrieval_worker", "merge_results")

    # Synthesizer -> Answer Generation
    builder.add_edge("merge_results", "answer_generation")

    # Answer -> Evaluation
    builder.add_edge("answer_generation", "evaluate_answer")

    # Evaluation -> Retry or End
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "answer_generation": "answer_generation",
            END: END,
        }
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Export compiled graph
multi_agent_rag_graph = build_multi_agent_rag_graph()
