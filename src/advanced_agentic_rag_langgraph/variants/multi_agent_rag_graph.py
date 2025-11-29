"""
Multi-Agent RAG Graph - Orchestrator-Worker Pattern.

For complex queries requiring multi-faceted retrieval.
Decomposes query into sub-queries, parallel worker retrieval, LLM relevance scoring.

Architecture: Workers inherit Advanced's core retrieval algorithms while being
optimized for parallel execution. The orchestration layer adds complexity
classification, query decomposition, and cross-agent fusion.

Core Retrieval Algorithms (inherited from Advanced):
1. Semantic vector search
2. Query expansion (always applied - sub-queries already focused by decomposition)
3. Hybrid retrieval (semantic + BM25)
4. RRF fusion (within each worker)
5. CrossEncoder reranking
6. LLM-as-judge reranking (two-stage)
7. Retrieval quality evaluation
8. Query rewriting loop (max 2 attempts per worker)
9. LLM-based strategy selection

Optimized for Parallel Execution (simplified from Advanced):
- No expansion decision LLM (sub-queries are focused by decomposition)
- No strategy revert validation (rare with 2-attempt max)
- Reduced logging (no ground truth tracking per worker)

Orchestration Features (main graph level):
10. Conversational query rewriting
11. Answer quality evaluation (5 issue types)
12. Adaptive thresholds (65%/50%)
13. Generation retry loop (adaptive temperature)
14. NLI-based hallucination detection
15. Refusal detection
16. Conversation context preservation

Multi-Agent Specific (+3 capabilities):
17. Complexity classification (simple vs complex routing)
18. Query decomposition (2-4 sub-queries)
19. Parallel worker retrieval with LLM relevance scoring

Graph Structure: 7 nodes, orchestrator-worker pattern
- conversational_rewrite_node
- classify_complexity_node (orchestrator decision)
- decompose_query_node (orchestrator)
- retrieval_subagent (parallel workers via Send API)
- merge_results_node (synthesizer with LLM relevance scoring)
- answer_generation_node
- evaluate_answer_node

Routing Functions:
- assign_workers: Fan-out to parallel retrieval workers
- route_after_evaluation: Generation retry or end

Pattern: Orchestrator-Worker (LangGraph docs)
https://docs.langchain.com/oss/python/langgraph/workflows-agents

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

from typing import TypedDict, Optional, Literal, Annotated, Union
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from advanced_agentic_rag_langgraph.utils.env import is_langgraph_api_environment
from advanced_agentic_rag_langgraph.retrieval import expand_query, rewrite_query
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
from advanced_agentic_rag_langgraph.retrieval.query_optimization import optimize_query_for_strategy
from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter
from advanced_agentic_rag_langgraph.prompts import get_prompt
from advanced_agentic_rag_langgraph.prompts.answer_generation import get_answer_generation_prompts
from advanced_agentic_rag_langgraph.validation import NLIHallucinationDetector
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_ndcg
from advanced_agentic_rag_langgraph.retrieval.multi_agent_merge_reranker import MultiAgentMergeReRanker

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
    k_final: Optional[int]  # 4 for standard dataset, 6 for hard dataset

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

DEFAULT BIAS (Important):
- Prefer SIMPLE unless the query has clear opportunity for parallel retrieval from distinct sources
- "Explain X in detail" or "How does X work?" = SIMPLE, even if X has many components
- A single focused retrieval can get all relevant sections from ONE source
- Decomposition adds overhead - only beneficial when comparing DIFFERENT entities

COMPLEX INDICATORS (decomposition beneficial):
- Comparative questions ("Compare X and Y", "differences between", "X vs Y")
- Cross-source synthesis (information must come from multiple distinct documents)
- Questions with explicit conjunctions across topics ("How does A relate to B")

SIMPLE INDICATORS (single retrieval sufficient):
- Single-source deep dives ("Explain X", "How does X work", "What are the components of X")
- Procedural questions with linear steps ("How do I do X?")
- Factual lookups ("What is X?", "Define Y", "When was X published?")
- Questions about ONE concept, even if detailed or multi-part

KEY DISTINCTION:
- "Explain the complete architecture of X" = SIMPLE (one topic, deep dive)
- "Compare how X and Y approach the same problem" = COMPLEX (two entities, explicit comparison)
- "What are all the components of X and how do they interact" = SIMPLE (still one topic)
- "How does X in system A differ from X in system B" = COMPLEX (cross-system comparison)

Return is_complex=True ONLY if decomposition would clearly improve retrieval by enabling parallel search across distinct sources."""

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


def route_after_complexity(state: MultiAgentRAGState) -> Union[Literal["decompose_query"], list[Send]]:
    """Route based on complexity: complex -> decompose, simple -> direct to workers."""
    if state.get("is_complex_query", True):
        return "decompose_query"
    # Simple query: skip decomposition, fan out directly to single worker
    return assign_workers(state)


# ========== QUERY DECOMPOSITION (ORCHESTRATOR) ==========

def decompose_query_node(state: MultiAgentRAGState) -> dict:
    """
    Decompose complex query into focused sub-queries (2 preferred, 3-4 if necessary).

    ORCHESTRATOR in the orchestrator-worker pattern.
    Each sub-query targets a distinct SOURCE for parallel retrieval.
    """
    question = state.get("baseline_query", state.get("user_question", ""))

    spec = get_model_for_task("query_decomposition")
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
    )
    structured_llm = llm.with_structured_output(QueryDecomposition)

    prompt = f"""Decompose this query into focused sub-queries for parallel retrieval.

Query: "{question}"

DECOMPOSITION PRINCIPLE:
- Use the MINIMUM number of sub-queries needed (2 is ideal, 3-4 only if truly necessary)
- Each sub-query should target a DIFFERENT SOURCE/DOCUMENT, not just a different aspect
- If aspects can be answered from the SAME source, keep them in ONE sub-query

WHEN TO USE 2 SUB-QUERIES (preferred):
- "Compare X and Y" -> [query about X, query about Y]
- "How does A differ from B" -> [query about A, query about B]

WHEN TO USE 3-4 SUB-QUERIES (only if necessary):
- Cross-system comparisons spanning 3+ distinct entities
- Questions explicitly requiring synthesis from 3+ different sources

GUIDELINES:
1. Each sub-query targets a DISTINCT source/document
2. Sub-queries are SELF-CONTAINED (understandable without original)
3. Sub-queries are PARALLEL (not dependent on each other)
4. Preserve technical terms exactly

EXAMPLES:
- "Compare X and Y approaches" ->
  ["What is the approach used in X?",
   "What is the approach used in Y?"]
  (2 sub-queries: each targets a different source)

- "How has technique T evolved across systems A, B, C, and D?" ->
  ["How does T work in A?",
   "How did B adapt T?",
   "How does C apply T?",
   "How does D use T?"]
  (4 sub-queries: each targets a different system/source)

Return the minimum number of sub-queries needed."""

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
    Falls back to baseline_query for simple queries that skip decomposition.
    """
    sub_queries = state.get("sub_queries") or [state.get("baseline_query", state.get("user_question", ""))]
    corpus_stats = state.get("corpus_stats", {})

    print(f"\n{'='*60}")
    print(f"ASSIGN WORKERS (Send API)")
    print(f"Spawning {len(sub_queries)} parallel retrieval workers")
    print(f"{'='*60}\n")

    return [
        Send("retrieval_subagent", {
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

    Flow: decide_strategy -> query_expansion -> retrieve_with_expansion -> [quality check]
                                                               |
                                            [>= 0.6 OR attempts >= 2] -> END
                                            [< 0.6 AND attempts < 2] -> rewrite -> query_expansion
    """
    builder = StateGraph(RetrievalSubgraphState)

    # Nodes
    builder.add_node("decide_strategy", _subgraph_decide_strategy_node)
    builder.add_node("query_expansion", _subgraph_query_expansion_node)
    builder.add_node("retrieve_with_expansion", _subgraph_retrieve_node)
    builder.add_node("rewrite_and_refine", _subgraph_rewrite_node)

    # Flow
    builder.add_edge(START, "decide_strategy")
    builder.add_edge("decide_strategy", "query_expansion")
    builder.add_edge("query_expansion", "retrieve_with_expansion")

    builder.add_conditional_edges(
        "retrieve_with_expansion",
        _subgraph_route_after_retrieval,
        {
            END: END,
            "rewrite_and_refine": "rewrite_and_refine",
        }
    )

    builder.add_edge("rewrite_and_refine", "query_expansion")

    return builder.compile()


def _subgraph_route_after_retrieval(state: RetrievalSubgraphState) -> str:
    """Pure router for subgraph retrieval quality check."""
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)

    # Max 2 attempts (aligned with main graph)
    if quality >= 0.6 or attempts >= 2:
        return END
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


# Build subgraph once
retrieval_subgraph = _build_retrieval_subgraph()


# ========== RETRIEVAL WORKER NODE ==========

def retrieval_subagent(state: WorkerState) -> dict:
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

        docs = result.get("retrieved_docs", [])
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
    Merge results from all parallel sub-agents using LLM relevance scoring.

    SYNTHESIZER in the orchestrator-worker pattern.
    Single-stage merge: LLM scores each document 0-100 by relevance to the
    original question, then sorts and takes top-k.

    Note: No RRF fusion - RRF rewards docs appearing in multiple workers, but
    decomposed sub-queries target DISTINCT sources, so overlap indicates
    generic content, not relevance.
    """
    sub_agent_results = state.get("sub_agent_results", [])
    original_question = state.get("baseline_query", state.get("user_question", ""))

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

    # ========== Single worker fast path ==========
    if len(sub_agent_results) == 1:
        # Single worker - docs already reranked by worker, just pass through
        result = sub_agent_results[0]
        docs = result.get("docs", [])
        k_final = state.get("k_final", 4)
        top_docs = docs[:k_final]
        quality_score = result.get("quality_score", 0.0)

        print(f"Single worker: Taking top-{k_final} of {len(docs)} docs directly (skip merge)")

        # Ground truth tracking
        ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])
        if ground_truth_doc_ids:
            final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in top_docs]
            found = [cid for cid in ground_truth_doc_ids if cid in final_chunk_ids]
            missing = [cid for cid in ground_truth_doc_ids if cid not in final_chunk_ids]
            print(f"Expected chunks: Found {found if found else '[]'} | Missing {missing if missing else '[]'}")

        docs_text = "\n---\n".join([
            f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc in top_docs
        ])

        print(f"{'='*60}\n")
        return {
            "retrieved_docs": [docs_text],
            "unique_docs_list": top_docs,
            "retrieval_quality_score": quality_score,
            "multi_agent_metrics": {
                "workers": 1,
                "total_unique_docs": len(docs),
                "multi_agent_docs": 0,
                "top_k_selected": len(top_docs),
                "avg_quality": quality_score,
                "merge_method": "single_worker_passthrough",
            },
        }

    # ========== Collect unique docs (deduplicate by doc_id) ==========
    doc_objects = {}
    doc_agent_count = {}  # Track which docs appear in multiple agents

    for result in sub_agent_results:
        worker_idx = result.get("worker_index", 0)
        docs = result.get("docs", [])

        for doc in docs:
            doc_id = doc.metadata.get("id", doc.page_content[:50])

            if doc_id not in doc_objects:
                doc_objects[doc_id] = doc
                doc_agent_count[doc_id] = set()

            doc_agent_count[doc_id].add(worker_idx)

    # Cap at 24 unique docs for LLM selection (6 x 4 workers for hard dataset)
    candidate_docs = list(doc_objects.values())[:24]

    # k_final from state (4 for standard, 6 for hard dataset)
    k_final = state.get("k_final", 4)

    print(f"Deduplication: {sum(len(r.get('docs', [])) for r in sub_agent_results)} total -> {len(doc_objects)} unique -> {len(candidate_docs)} candidates")

    # Ground truth tracking
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])
    if ground_truth_doc_ids:
        candidate_chunk_ids = [doc.metadata.get("id", "unknown") for doc in candidate_docs]
        found = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in candidate_chunk_ids]
        missing = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in candidate_chunk_ids]
        print(f"Expected chunks: Found {found if found else '[]'} | Missing {missing if missing else '[]'}")

    # ========== LLM relevance scoring (multi-worker only, single worker returns early) ==========
    reranker = MultiAgentMergeReRanker(top_k=k_final)
    top_docs = reranker.rerank(
        original_question=original_question,
        candidate_docs=candidate_docs,
    )
    print(f"LLM Scoring: {len(candidate_docs)} candidates -> {len(top_docs)} selected")

    # Ground truth tracking after LLM relevance scoring
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
        "merge_method": "llm_relevance_scoring",
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
    builder.add_node("retrieval_subagent", retrieval_subagent)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)

    # ========== FLOW ==========
    builder.add_edge(START, "conversational_rewrite")
    builder.add_edge("conversational_rewrite", "classify_complexity")

    # Complexity routing: complex -> decompose, simple -> direct to workers
    builder.add_conditional_edges(
        "classify_complexity",
        route_after_complexity,
        {
            "decompose_query": "decompose_query",
            "retrieval_subagent": "retrieval_subagent",
        }
    )

    # Orchestrator -> Workers (parallel via Send)
    builder.add_conditional_edges(
        "decompose_query",
        assign_workers,
        ["retrieval_subagent"]
    )

    # Workers -> Synthesizer
    builder.add_edge("retrieval_subagent", "merge_results")

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

    # Skip checkpointer when running under LangGraph API (provides its own persistence)
    checkpointer = None if is_langgraph_api_environment() else MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Export compiled graph
multi_agent_rag_graph = build_multi_agent_rag_graph()
