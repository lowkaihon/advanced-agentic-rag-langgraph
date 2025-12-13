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
13. Generation retry loop (structured feedback)
14. HHEM-based hallucination detection
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
from advanced_agentic_rag_langgraph.validation import HHEMHallucinationDetector
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_ndcg
from advanced_agentic_rag_langgraph.retrieval.multi_agent_merge_reranker import MultiAgentMergeReRanker

# ========== GLOBALS ==========

adaptive_retriever = None
conversational_rewriter = ConversationalRewriter()
strategy_selector = StrategySelector()
hhem_detector = HHEMHallucinationDetector()


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
    previous_answer: Optional[str]  # Previous answer for retry context (enables targeted correction)

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
    keywords_to_inject: Optional[list[str]]


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
    keywords_to_inject: list[str]


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
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity,
    )
    structured_llm = llm.with_structured_output(ComplexityDecision)

    prompt = f"""Classify if this query benefits from decomposition into focused sub-queries.

Query: "{question}"

SIMPLE (single retrieval sufficient):
- Single-aspect questions: "What is X?", "How does X work?"
- Focused lookups: "Define X", "When was X published?"
- Questions targeting ONE specific thing

COMPLEX (decomposition beneficial):
- Multi-aspect questions: "Explain X including A, B, and C" (3 aspects)
- Comparative questions: "Compare X and Y" (2 entities)
- Questions with multiple distinct retrieval targets

===== FEW-SHOT EXAMPLES =====

Query: "What is X?"
Classification: SIMPLE
Reasoning: Single aspect, single retrieval target.

Query: "How does X work?"
Classification: SIMPLE
Reasoning: Single focused question about one thing.

Query: "Explain X, including A, B, and C."
Classification: COMPLEX
Reasoning: Three distinct aspects (A, B, C) benefit from focused sub-queries.

Query: "Compare X and Y."
Classification: COMPLEX
Reasoning: Two entities = two retrieval targets.

Query: "What are the differences between X and Y?"
Classification: COMPLEX
Reasoning: Comparing two things = decompose into one query per entity.

=============================

KEY RULE: Count the distinct aspects/entities.
- 1 aspect = SIMPLE
- 2+ aspects = COMPLEX"""

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
    Decompose complex query into focused sub-queries (one per aspect/entity).

    ORCHESTRATOR in the orchestrator-worker pattern.
    Sub-query count matches the number of distinct aspects in the original query.
    """
    question = state.get("baseline_query", state.get("user_question", ""))

    spec = get_model_for_task("query_decomposition")
    llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity,
    )
    structured_llm = llm.with_structured_output(QueryDecomposition)

    prompt = f"""Decompose this query into focused sub-queries (one per aspect/entity).

Query: "{question}"

DECOMPOSITION RULE:
- One sub-query per distinct aspect or entity mentioned
- Count aspects explicitly mentioned in the query

===== FEW-SHOT EXAMPLES =====

Query: "Explain X, including A, B, and C."
Sub-queries: ["What is A in X?", "What is B in X?", "What is C in X?"]
Reasoning: Three aspects (A, B, C) = 3 sub-queries.

Query: "Compare X and Y."
Sub-queries: ["How does X work?", "How does Y work?"]
Reasoning: Two entities = 2 sub-queries.

Query: "What are the differences between X and Y?"
Sub-queries: ["What is X?", "What is Y?"]
Reasoning: Two entities = 2 sub-queries.

=============================

GUIDELINES:
1. One sub-query per aspect/entity (match the count in the original query)
2. Sub-queries are SELF-CONTAINED
3. Preserve technical terms exactly"""

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
                                          ^                    |
                                          |   [>= 0.6 OR attempts >= 2] -> END
                                          |   [off_topic/wrong_domain] -> query_expansion (strategy switch)
                                          +-- [other issues] -> rewrite -> query_expansion
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
            "query_expansion": "query_expansion",
            "rewrite_and_refine": "rewrite_and_refine",
        }
    )

    builder.add_edge("rewrite_and_refine", "query_expansion")

    return builder.compile()


def _subgraph_route_after_retrieval(state: RetrievalSubgraphState) -> str:
    """Pure router for subgraph retrieval quality check."""
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])

    # Max 2 attempts (aligned with main graph)
    if quality >= 0.6 or attempts >= 2:
        return END

    # Early strategy switch path (off_topic/wrong_domain on first attempt)
    if ("off_topic" in issues or "wrong_domain" in issues) and attempts == 1:
        return "query_expansion"

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
    """Generate strategy-agnostic expansions, then optimize expansions[0] for strategy."""
    query = state.get("active_query", state["sub_query"])
    current_strategy = state.get("retrieval_strategy", "hybrid")

    quality = state.get("retrieval_quality_score", 1.0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])

    # Check for early strategy switch (same logic as nodes.py)
    early_switch = (quality < 0.6 and
                    attempts == 1 and
                    ("off_topic" in issues or "wrong_domain" in issues))

    old_strategy = None
    strategy_updates = {}

    if early_switch:
        old_strategy = current_strategy
        next_strategy = "keyword" if current_strategy != "keyword" else "hybrid"

        print(f"  [Worker] EARLY STRATEGY SWITCH: {current_strategy} -> {next_strategy}")
        print(f"  [Worker] Reason: {', '.join(issues)}")

        current_strategy = next_strategy
        strategy_updates = {
            "retrieval_strategy": next_strategy,
        }

    # 1. Expand FIRST (strategy-agnostic variants for RRF diversity)
    expansions = expand_query(query)

    # 2. Optimize for strategy
    optimized_query = optimize_query_for_strategy(
        query=query,
        strategy=current_strategy,
        old_strategy=old_strategy,
        issues=issues if early_switch else []
    )

    # 3. Replace expansions[0] with optimized version
    expansions[0] = optimized_query

    return {
        "retrieval_query": optimized_query,
        "query_expansions": expansions,
        **strategy_updates,
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
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity,
    )
    structured_llm = quality_llm.with_structured_output(RetrievalQualityEvaluation)

    quality_prompt = get_prompt("retrieval_quality_eval", query=query, docs_text=docs_text)

    try:
        evaluation = structured_llm.invoke(quality_prompt)
        quality_score = evaluation["quality_score"] / 100
        quality_issues = evaluation["issues"]
        keywords_to_inject = evaluation.get("keywords_to_inject", [])
    except Exception as e:
        print(f"  [Worker] Quality evaluation failed: {e}")
        quality_score = 0.5
        quality_issues = []
        keywords_to_inject = []

    attempts = state.get("retrieval_attempts", 0) + 1
    print(f"  [Worker] Retrieved {len(final_docs)} docs, quality: {quality_score:.0%}, attempt: {attempts}/2")

    return {
        "retrieved_docs": final_docs,
        "retrieval_quality_score": quality_score,
        "retrieval_quality_issues": quality_issues,
        "keywords_to_inject": keywords_to_inject,
        "retrieval_attempts": attempts,
    }


def _subgraph_rewrite_node(state: RetrievalSubgraphState) -> dict:
    """Inject diagnostic-suggested keywords into query for improved retrieval."""
    query = state.get("active_query", state["sub_query"])
    keywords = state.get("keywords_to_inject", [])

    if not keywords:
        return {"active_query": query}

    refined_query = rewrite_query(query, keywords)
    print(f"  [Worker] Keywords injected: {query[:50]}... -> {refined_query[:50]}...")

    return {
        "active_query": refined_query,
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

    # ========== Set-wise coverage selection ==========
    # Multi-agent merge only runs for complex queries (simple queries use single-worker fast path)
    sub_queries = state.get("sub_queries", [])

    reranker = MultiAgentMergeReRanker(top_k=k_final)
    top_docs, _ = reranker.rerank(
        original_question=original_question,
        candidate_docs=candidate_docs,
        sub_queries=sub_queries,
    )
    print(f"LLM Scoring: {len(candidate_docs)} candidates -> {len(top_docs)} selected")

    # Ground truth tracking after LLM relevance scoring
    if ground_truth_doc_ids:
        final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in top_docs]
        found_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in final_chunk_ids]
        missing_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in final_chunk_ids]
        print(f"\nExpected chunks in final selection:")
        print(f"Found: {found_in_final if found_in_final else '[]'} | Missing: {missing_in_final if missing_in_final else '[]'}")

    # Calculate average quality from worker scores (set_selection doesn't produce scores)
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
        "merge_method": "set_selection",
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
    """
    Generate answer from merged multi-agent context.

    Implements RAG best practices: quality-aware thresholds, XML markup, unified feedback.
    Handles both initial generation and retries from combined evaluation.
    """
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
            "final_answer": "I apologize, but I could not retrieve any relevant documents to answer your question. Please try rephrasing your query or check if the information exists in the knowledge base.",
            "generation_attempts": generation_attempts,
            "messages": [AIMessage(content="Empty retrieval - no answer generated")],
        }

    formatted_context = context

    # Determine quality instruction and retry feedback based on scenario
    # LLM best practices: System prompt = behavioral, User message = content
    retry_feedback_content = ""  # Goes to user message via retry_feedback param

    if generation_attempts > 1 and retry_feedback:
        # RETRY MODE: Split behavioral guidance (system) from content (user message)
        previous_answer = state.get("previous_answer", "")

        # Behavioral guidance only (goes to system prompt)
        quality_instruction = "RETRY: Focus on fixing the issues described in <retry_instructions>. Prioritize factual accuracy over comprehensiveness."

        # Content with previous answer + issues (goes to user message)
        retry_feedback_content = f"""Your previous answer was:
---
{previous_answer}
---

Issues with your previous answer:
{retry_feedback}

Generate an improved answer that fixes ALL issues above. Do not repeat the same unsupported claims."""

        print(f"RETRY MODE:")
        print(f"Feedback:\n{retry_feedback}\n")
    else:
        # First generation - use quality-aware instructions
        if retrieval_quality >= 0.8:
            quality_instruction = f"""High Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents are highly relevant and should contain the information needed to answer the question. Answer directly and confidently based on them."""
        elif retrieval_quality >= 0.6:
            quality_instruction = f"""Medium Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents are somewhat relevant but may have gaps in coverage. Use them to answer what you can, but explicitly acknowledge any limitations or missing information."""
        else:
            quality_instruction = f"""Low Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents may not fully address the question. Only answer what can be directly supported by the context. If the context is insufficient, clearly state: "The provided context does not contain enough information to answer this question completely." """

    spec = get_model_for_task("answer_generation")
    is_gpt5 = spec.name.lower().startswith("gpt-5")

    # System prompt = behavioral guidance, User message = content + retry feedback
    system_prompt, user_message = get_answer_generation_prompts(
        quality_instruction=quality_instruction,
        formatted_context=formatted_context,
        question=question,
        is_gpt5=is_gpt5,
        retry_feedback=retry_feedback_content,  # Content with previous answer + issues
    )

    # Flat low temperature for groundedness preservation
    # Quality improvements come from retry_feedback prompt guidance, not temperature randomness
    # (Variable temp schedule removed after HHEM testing showed 0.7 hurt groundedness)
    temperature = 0.3

    llm = ChatOpenAI(
        model=spec.name,
        temperature=temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity,
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
        "incomplete_synthesis": "Provide more comprehensive synthesis of the relevant information",
        "lacks_specificity": "Include specific details (numbers, dates, names, technical terms)",
        "wrong_focus": "Re-read question and address the primary intent",
        "partial_answer": "Ensure all question parts are answered completely",
        "missing_details": "Add more depth and explanation where the context provides supporting information",
    }
    return "; ".join([guidance_map.get(issue, issue) for issue in issues])


def evaluate_answer_node(state: MultiAgentRAGState) -> dict:
    """
    Combined refusal detection + groundedness + quality evaluation (single decision point).

    Performs checks in sequence (with early exit on refusal):
    1. Refusal detection (LLM-as-judge) - CHECK FIRST, exit early if refusal
    2. HHEM-based hallucination detection (factuality)
    3. LLM-as-judge quality assessment (sufficiency)

    Returns unified decision: is answer good enough to return?
    """
    answer = state.get("final_answer", "")
    # Extract individual chunks for per-chunk HHEM verification (stays under 512 token limit)
    unique_docs = state.get("unique_docs_list", [])
    chunks = [doc.page_content for doc in unique_docs] if unique_docs else []
    question = state.get("baseline_query", state.get("user_question", ""))
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_attempts = state.get("generation_attempts", 0)

    print(f"\n{'='*60}")
    print(f"ANSWER EVALUATION (Refusal + Groundedness + Quality)")
    print(f"Generation attempt: {generation_attempts}")
    print(f"Retrieval quality: {retrieval_quality:.0%}")

    # ==== 1. REFUSAL DETECTION (LLM-as-judge) - Check FIRST ====

    # Detect if LLM refused to answer due to insufficient context
    refusal_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,  # Deterministic for classification
    )
    refusal_checker = refusal_llm.with_structured_output(RefusalCheck)

    try:
        refusal_result = refusal_checker.invoke([{
            "role": "system",
            "content": """You are evaluating whether an AI assistant FULLY REFUSED to answer a question due to insufficient context.

IMPORTANT DISTINCTION:
- FULL REFUSAL: No substantive answer provided, only acknowledges insufficiency (e.g., "The provided context does not contain enough information to answer this question.")
- PARTIAL ANSWER: Provides some information from context BUT acknowledges limitations/gaps (e.g., "Based on the documents, X is true, though the context doesn't provide information about Y.")

The assistant was instructed to use this phrase for FULL REFUSALS:
"The provided context does not contain enough information to answer this question."

ONLY flag as refusal if the answer provides NO substantive information."""
        }, {
            "role": "user",
            "content": f"""Question: {question}

Answer: {answer}

Determine if the assistant FULLY REFUSED (provided NO answer):

FULL REFUSAL indicators (return refused=True):
- Answer provides NO substantive information at all
- Only acknowledges that context is insufficient without providing any facts
- Examples: "I cannot answer this question", "The context doesn't contain this information"

PARTIAL ANSWER indicators (return refused=False):
- Contains ANY facts, details, or explanations from the context
- Even if it acknowledges gaps (e.g., "X uses dropout... however, the learning rate schedule is not covered")
- Provides some useful information even if incomplete

CRITICAL: Presence of limitation phrases does NOT make it a refusal if substantive information is also provided.

Return:
- refused: True ONLY if complete refusal with NO substantive answer
- reasoning: Explain whether answer provided substantive information or only acknowledged insufficiency"""
        }])
        is_refusal = refusal_result["refused"]
        refusal_reasoning = refusal_result["reasoning"]
    except Exception as e:
        print(f"Warning: Refusal detection failed: {e}. Defaulting to not refused.")
        is_refusal = False
        refusal_reasoning = f"Detection failed: {e}"

    print(f"Refusal detection: {'REFUSED' if is_refusal else 'ATTEMPTED'} - {refusal_reasoning}")

    # Early exit if refusal detected (skip expensive checks)
    if is_refusal:
        print(f"Skipping groundedness and quality checks (refusal detected)")
        print(f"{'='*60}\n")
        return {
            "is_refusal": True,  # Terminal state
            "is_answer_sufficient": False,
            "groundedness_score": 1.0,  # Refusal is perfectly grounded (no unsupported claims)
            "has_hallucination": False,
            "unsupported_claims": [],
            "confidence_score": 0.0,
            "answer_quality_reasoning": "Evaluation skipped (LLM refused to answer)",
            "answer_quality_issues": [],
            "retry_feedback": "",
            "messages": [AIMessage(content=f"Refusal detected: {refusal_reasoning}")],
        }

    # ==== 2. GROUNDEDNESS CHECK (HHEM) ====

    # Run HHEM groundedness check with per-chunk verification
    groundedness_result = hhem_detector.verify_groundedness(answer, chunks)
    groundedness_score = groundedness_result.get("groundedness_score", 1.0)
    has_hallucination = groundedness_score < 0.8
    unsupported_claims = groundedness_result.get("unsupported_claims", [])
    print(f"Groundedness: {groundedness_score:.0%}")

    # ==== 3. QUALITY CHECK (LLM-as-judge) ====

    retrieval_quality_issues = state.get("retrieval_quality_issues", [])
    has_missing_info = any(issue in retrieval_quality_issues for issue in ["partial_coverage", "missing_key_info", "incomplete_context"])
    quality_threshold = 0.5 if (retrieval_quality < 0.6 or has_missing_info) else 0.65

    spec = get_model_for_task("answer_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity,
    )
    structured_llm = quality_llm.with_structured_output(AnswerQualityEvaluation)

    evaluation_prompt = get_prompt(
        "answer_quality_eval",
        question=question,
        answer=answer,
        retrieval_quality=f"{retrieval_quality:.0%}",
        retrieval_issues=', '.join(retrieval_quality_issues) if retrieval_quality_issues else 'None',
        quality_threshold_pct=quality_threshold*100,
        quality_threshold_low_pct=(quality_threshold-0.15)*100,
        quality_threshold_minus_1_pct=quality_threshold*100-1,
        quality_threshold_low_minus_1_pct=(quality_threshold-0.15)*100-1
    )

    try:
        evaluation = structured_llm.invoke(evaluation_prompt)
        confidence = evaluation["confidence_score"] / 100
        reasoning = evaluation["reasoning"]
        quality_issues = evaluation["issues"]
    except Exception as e:
        print(f"Warning: Answer evaluation failed: {e}. Using conservative fallback.")
        evaluation = {
            "is_relevant": True,
            "is_complete": False,
            "is_accurate": True,
            "confidence_score": 50,
            "reasoning": f"Evaluation failed: {e}",
            "issues": ["evaluation_error"]
        }
        confidence = 0.5
        reasoning = evaluation["reasoning"]
        quality_issues = evaluation["issues"]

    is_quality_sufficient = (
        evaluation["is_relevant"] and
        evaluation["is_complete"] and
        evaluation["is_accurate"] and
        confidence >= quality_threshold
    )

    print(f"Quality: {confidence:.0%} ({'sufficient' if is_quality_sufficient else 'insufficient'})")
    if quality_issues:
        print(f"Issues: {', '.join(quality_issues)}")

    # ==== 4. COMBINED DECISION ====

    has_issues = has_hallucination or not is_quality_sufficient

    # Build unified feedback for retry - PRIORITIZE groundedness over quality
    retry_feedback_parts = []
    if has_hallucination:
        # Hallucination detected: Only give grounding feedback (ignore quality issues)
        # Rationale: Quality can't improve until grounding is fixed; mixed feedback is contradictory
        retry_feedback_parts.append(
            f"HALLUCINATION DETECTED ({groundedness_score:.0%} grounded):\n"
            f"Unsupported claims: {', '.join(unsupported_claims)}\n"
            f"Fix: ONLY state facts explicitly in retrieved context. If information is missing, acknowledge the limitation rather than adding unsupported details."
        )
    elif not is_quality_sufficient:
        # No hallucination, but quality issues: Safe to push for improvements
        retry_feedback_parts.append(
            f"QUALITY ISSUES:\n"
            f"Problems: {', '.join(quality_issues)}\n"
            f"Fix: {_get_quality_fix_guidance(quality_issues)}"
        )

    retry_feedback = "\n\n".join(retry_feedback_parts) if retry_feedback_parts else ""

    print(f"Combined decision: {'RETRY' if has_issues else 'SUFFICIENT'}")
    print(f"{'='*60}\n")

    return {
        "is_answer_sufficient": not has_issues,
        "groundedness_score": groundedness_score,
        "has_hallucination": has_hallucination,
        "unsupported_claims": unsupported_claims,
        "confidence_score": confidence,
        "answer_quality_reasoning": reasoning,
        "answer_quality_issues": quality_issues,
        "retry_feedback": retry_feedback,
        "previous_answer": answer if has_issues else None,  # Store for retry context
        "is_refusal": False,  # Only reached for non-refusals (early exit handles refusals)
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
    if generation_attempts < 2:  # reduce from 3 to 2 to quicken test
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
