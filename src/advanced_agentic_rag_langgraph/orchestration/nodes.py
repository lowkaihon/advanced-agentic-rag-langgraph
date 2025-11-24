from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from advanced_agentic_rag_langgraph.retrieval import (
    expand_query,
    rewrite_query,
    AdaptiveRetriever,
    LLMMetadataReRanker,
    SemanticRetriever,
)
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_ndcg
from advanced_agentic_rag_langgraph.validation import NLIHallucinationDetector
from advanced_agentic_rag_langgraph.retrieval.query_optimization import optimize_query_for_strategy
from advanced_agentic_rag_langgraph.prompts import get_prompt
from advanced_agentic_rag_langgraph.prompts.answer_generation import get_answer_generation_prompts
import re
import json


adaptive_retriever = None
conversational_rewriter = ConversationalRewriter()
strategy_selector = StrategySelector()
nli_detector = NLIHallucinationDetector()


# ============ STRUCTURED OUTPUT SCHEMAS ============

class ExpansionDecision(TypedDict):
    """Structured output schema for query expansion decision.

    Used with .with_structured_output() to ensure reliable LLM parsing.
    """
    decision: Literal["yes", "no"]
    reasoning: str


class RetrievalQualityEvaluation(TypedDict):
    """Structured output schema for retrieval quality assessment.

    Used with .with_structured_output() to ensure reliable LLM parsing.
    """
    quality_score: float
    reasoning: str
    issues: list[str]


class AnswerQualityEvaluation(TypedDict):
    """Structured output schema for answer quality assessment.

    Mirrors RetrievalQualityEvaluation pattern for consistency.
    """
    is_relevant: bool
    is_complete: bool
    is_accurate: bool
    confidence_score: float
    reasoning: str
    issues: list[str]


# ========== HELPER FUNCTIONS ==========

def _get_answer_generation_llm():
    """Get LLM for answer generation with tier-based configuration."""
    spec = get_model_for_task("answer_generation")
    return ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )


# ============ CONVERSATIONAL QUERY REWRITING ============

def extract_conversation_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """
    Extract conversation history from messages list (LangGraph best practice).

    Pairs HumanMessage/AIMessage to create conversation turns in the format
    expected by ConversationalRewriter: [{"user": str, "assistant": str}, ...]

    Only includes complete pairs (ignores trailing unpaired messages).

    Args:
        messages: List of BaseMessage objects (HumanMessage, AIMessage, etc.)

    Returns:
        List of conversation turns in format: [{"user": str, "assistant": str}]

    Example:
        >>> messages = [
        ...     HumanMessage(content="What is RAG?"),
        ...     AIMessage(content="RAG is Retrieval-Augmented Generation..."),
        ...     HumanMessage(content="How does it work?"),
        ...     AIMessage(content="It works by...")
        ... ]
        >>> extract_conversation_history(messages)
        [
            {"user": "What is RAG?", "assistant": "RAG is..."},
            {"user": "How does it work?", "assistant": "It works by..."}
        ]
    """
    if not messages or len(messages) < 2:
        return []

    conversation = []
    i = 0

    while i < len(messages) - 1:
        # Look for HumanMessage followed by AIMessage
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i+1], AIMessage):
            conversation.append({
                "user": messages[i].content,
                "assistant": messages[i+1].content
            })
            i += 2
        else:
            i += 1

    return conversation


def conversational_rewrite_node(state: dict) -> dict:
    """
    Rewrite query using conversation history to make it self-contained.

    This node runs before query expansion to ensure queries have proper context.
    Extracts conversation from messages field (LangGraph best practice).
    """
    question = state.get("user_question", "")

    # Extract conversation from messages (LangGraph best practice)
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
        print(f"Reasoning: {reasoning}")
        print(f"Conversation turns used: {len(conversation_history)}")
        print(f"State transition: user_question={question}, baseline_query={rewritten_query}")
        print(f"{'='*60}\n")

    return {
        "baseline_query": rewritten_query,
        "active_query": rewritten_query,
        "corpus_stats": get_corpus_stats(),
        "messages": [HumanMessage(content=question)],
        "retrieval_attempts": 0,  # Reset counter for new user question (fixes multi-turn conversation bug)
        "generation_retry_count": 0,  # Reset for new user question (unified retry counter)
        "retry_feedback": None,  # Clear feedback for new user question
    }

# ============ QUERY OPTIMIZATION STAGE ============

def _should_skip_expansion_llm(query: str) -> bool:
    """
    Use LLM to determine if query expansion would improve retrieval.

    Domain-agnostic - works for any query type and corpus.
    More accurate than heuristics - handles context and intent.
    """
    spec = get_model_for_task("expansion_decision")
    expansion_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )

    prompt = f"""Should this query be expanded into multiple variations for better retrieval?

Query: "{query}"

EXPANSION IS BENEFICIAL FOR:
- Ambiguous queries that could be phrased multiple ways
- Complex questions where synonyms/related terms would help
- Queries where users might use different terminology
- Conceptual questions with multiple valid phrasings

SKIP EXPANSION FOR:
- Clear, specific queries that are already well-formed
- Simple factual lookups (definitions, direct questions)
- Queries with exact-match intent only (pure lookups)
- Procedural queries with specific steps (expansion adds noise)
- Queries that are already precise and unambiguous

IMPORTANT CONSIDERATIONS:
- Consider overall intent, not just query length or presence of quotes
- Quoted terms don't automatically mean skip - consider if variations help
- Example: "Compare 'X' and 'Y'" has quotes BUT expansion helps (synonyms for "compare")
- Example: "What is Z?" is simple BUT expansion might help (rephrasing)

Return your decision ('yes' or 'no') with brief reasoning."""

    try:
        structured_llm = expansion_llm.with_structured_output(ExpansionDecision)
        result = structured_llm.invoke(prompt)
        skip = (result["decision"] == "no")

        print(f"\n{'='*60}")
        print(f"EXPANSION DECISION")
        print(f"Query: {query}")
        print(f"LLM decision: {'SKIP expansion' if skip else 'EXPAND query'}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"{'='*60}\n")

        return skip
    except Exception as e:
        print(f"Warning: Expansion decision LLM failed: {e}, defaulting to expand")
        return False


def query_expansion_node(state: dict) -> dict:
    """
    Optimize query for strategy, then conditionally expand.

    ALL queries passing through this node get strategy-specific optimization.
    This consolidates optimization logic in a single location.

    Entry paths:
    1. Initial turn: From decide_strategy (first question) - optimizes for selected strategy
    2. Early switch: From route_after_retrieval (off_topic/wrong_domain detected) - switches strategy then optimizes
    3. Query rewrite: From rewrite_and_refine (semantic query improvement) - optimizes rewritten query

    NO LONGER HANDLES:
    - Late strategy switching (removed - no re-retrieval after answer_generation)
    - Hallucination-triggered re-retrieval (removed - unreachable code)
    """

    quality = state.get("retrieval_quality_score", 1.0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])
    current_strategy = state.get("retrieval_strategy", "hybrid")

    # Check for early strategy switch
    early_switch = (quality < 0.6 and
                    attempts == 1 and
                    ("off_topic" in issues or "wrong_domain" in issues))

    old_strategy = None
    strategy_updates = {}

    if early_switch:
        # Off-topic results indicate need for precision -> keyword search
        old_strategy = current_strategy
        next_strategy = "keyword" if current_strategy != "keyword" else "hybrid"

        print(f"\n{'='*60}")
        print(f"EARLY STRATEGY SWITCH")
        print(f"From: {current_strategy} to {next_strategy}")
        print(f"Reason: {', '.join(issues)}")
        print(f"{'='*60}\n")

        current_strategy = next_strategy  # Use new strategy for optimization

        strategy_updates = {
            "retrieval_strategy": next_strategy,
            "strategy_switch_reason": f"Early detection: {', '.join(issues)}",
            "strategy_changed": True,
        }

    # ALWAYS optimize query for current strategy (consolidated optimization logic)
    source_query = state.get("active_query", state["baseline_query"])

    optimized_query = optimize_query_for_strategy(
        query=source_query,
        strategy=current_strategy,
        old_strategy=old_strategy,  # Only set during early switch
        issues=issues if early_switch else []
    )

    # Decide whether to expand optimized query
    if _should_skip_expansion_llm(optimized_query):
        result = {
            "retrieval_query": optimized_query,
            "query_expansions": [optimized_query],
            **strategy_updates
        }
        return result

    # Expand optimized query
    expansions = expand_query(optimized_query)
    print(f"\n{'='*60}")
    print(f"QUERY EXPANDED")
    print(f"Optimized query: {optimized_query}")
    print(f"Expansions: {expansions[1:]}")
    print(f"{'='*60}\n")

    result = {
        "retrieval_query": optimized_query,
        "query_expansions": expansions,
        **strategy_updates
    }
    return result

def decide_retrieval_strategy_node(state: dict) -> dict:
    """
    Decide which retrieval strategy to use based on query and corpus characteristics.

    Uses pure LLM classification for intelligent, domain-agnostic strategy selection.
    Query optimization happens downstream in query_expansion_node (consolidates all optimization logic).
    """
    query = state["baseline_query"]
    corpus_stats = state.get("corpus_stats", {})

    strategy, confidence, reasoning = strategy_selector.select_strategy(
        query,
        corpus_stats
    )

    print(f"\n{'='*60}")
    print(f"STRATEGY SELECTION")
    print(f"Query: {query}")
    print(f"Selected: {strategy.upper()}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Reasoning: {reasoning}")
    print(f"Note: Query optimization will happen in query_expansion_node")
    print(f"{'='*60}\n")

    return {
        "retrieval_strategy": strategy,
        "messages": [AIMessage(content=f"Strategy: {strategy} (confidence: {confidence:.0%})")],
    }

# ============ ADAPTIVE RETRIEVAL STAGE ============

def retrieve_with_expansion_node(state: dict) -> dict:
    """
    Retrieve documents using query expansions with RRF (Reciprocal Rank Fusion).

    RRF aggregates rankings across multiple query variants to improve retrieval quality.
    Formula: score(doc) = sum(1/(rank + k)) across all queries where doc appears.
    Research shows 3-5% MRR improvement over naive deduplication.
    """

    global adaptive_retriever
    if adaptive_retriever is None:
        adaptive_retriever = setup_retriever()

    strategy = state.get("retrieval_strategy", "hybrid")

    doc_ranks = {}
    doc_objects = {}

    expansion_source = "retrieval_query" if state.get("retrieval_query") else "active_query"
    expansions_count = len(state.get("query_expansions", []))
    print(f"\n{'='*60}")
    print(f"RETRIEVAL EXECUTION START")
    print(f"Using {expansions_count} query expansion(s)")
    print(f"Expansions generated from: {expansion_source}")
    print(f"Retrieval strategy: {strategy}")
    print(f"{'='*60}\n")

    for query in state.get("query_expansions", []):
        docs = adaptive_retriever.retrieve_without_reranking(query, strategy=strategy)

        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
                doc_objects[doc_id] = doc
            doc_ranks[doc_id].append(rank)

    k = 60
    rrf_scores = {}
    for doc_id, ranks in doc_ranks.items():
        rrf_score = sum(1.0 / (rank + k) for rank in ranks)
        rrf_scores[doc_id] = rrf_score

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    unique_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]

    # Extract ground truth for debugging (if available)
    ground_truth_doc_ids = state.get("ground_truth_doc_ids", [])

    print(f"\n{'='*60}")
    print(f"RRF MULTI-QUERY RETRIEVAL")
    print(f"Query variants: {len(state['query_expansions'])}")
    print(f"Total retrievals: {sum(len(ranks) for ranks in doc_ranks.values())}")
    print(f"Unique docs after RRF: {len(unique_docs)}")

    # Show ALL chunk IDs with RRF scores (typically 16-22 chunks)
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

    reranking_input = unique_docs[:40]

    print(f"{'='*60}")
    print(f"TWO-STAGE RERANKING (After RRF)")
    print(f"Input: {len(reranking_input)} docs (from RRF top-40)")

    # Show chunk IDs going into reranking
    reranking_chunk_ids = [doc.metadata.get("id", "unknown") for doc in reranking_input]
    print(f"\nChunk IDs sent to reranking (top-40):")
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

    query_for_reranking = state.get('active_query', state['baseline_query'])

    query_source = "active_query" if state.get("active_query") else "baseline_query"
    query_type = "semantic, human-readable" if query_source == "active_query" else "conversational, self-contained"
    query_type_description = "semantic query" if query_source == "active_query" else "conversational query"
    print(f"\n{'='*60}")
    print(f"RERANKING QUERY SOURCE")
    print(f"Using: {query_source} ({query_type})")
    print(f"Query: {query_for_reranking}")
    print(f"Note: Reranking uses {query_type_description}, NOT algorithm-optimized retrieval_query")
    print(f"{'='*60}\n")

    ranked_results = adaptive_retriever.reranker.rank(
        query_for_reranking,
        reranking_input
    )

    unique_docs = [doc for doc, score in ranked_results]
    reranking_scores = [score for doc, score in ranked_results]

    print(f"\nOutput: {len(unique_docs)} docs after two-stage reranking")

    # Show final chunk IDs with reranking scores
    print(f"\nFinal chunk IDs (after two-stage reranking):")
    for i, (doc, score) in enumerate(zip(unique_docs, reranking_scores), 1):
        chunk_id = doc.metadata.get("id", "unknown")
        print(f"  {i}. {chunk_id} (score: {score:.4f})")

    # Track ground truth in final results
    if ground_truth_doc_ids:
        final_chunk_ids = [doc.metadata.get("id", "unknown") for doc in unique_docs]
        found_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id in final_chunk_ids]
        missing_in_final = [chunk_id for chunk_id in ground_truth_doc_ids if chunk_id not in final_chunk_ids]
        print(f"\nExpected chunks in final results:")
        print(f"Found: {found_in_final if found_in_final else '[]'} | Missing: {missing_in_final if missing_in_final else '[]'}")

    print(f"{'='*60}\n")

    docs_text = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in unique_docs
    ])

    spec = get_model_for_task("retrieval_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
    structured_quality_llm = quality_llm.with_structured_output(RetrievalQualityEvaluation)

    quality_prompt = get_prompt("retrieval_quality_eval", query=state.get('active_query', state['baseline_query']), docs_text=docs_text)

    try:
        evaluation = structured_quality_llm.invoke(quality_prompt)
        quality_score = evaluation["quality_score"] / 100
        quality_reasoning = evaluation["reasoning"]
        quality_issues = evaluation["issues"]
    except Exception as e:
        print(f"Warning: Quality evaluation failed: {e}. Using neutral score.")
        quality_score = 0.5
        quality_reasoning = "Evaluation failed"
        quality_issues = []

    retrieval_metrics = {}
    ground_truth_doc_ids = state.get("ground_truth_doc_ids")
    relevance_grades = state.get("relevance_grades")

    if ground_truth_doc_ids:
        retrieval_metrics = calculate_retrieval_metrics(
            unique_docs,
            ground_truth_doc_ids,
            k=adaptive_retriever.k_final
        )

        if relevance_grades:
            retrieval_metrics["ndcg_at_k"] = calculate_ndcg(
                unique_docs,
                relevance_grades,
                k=adaptive_retriever.k_final
            )

        k = adaptive_retriever.k_final
        print(f"\n{'='*60}")
        print(f"RETRIEVAL METRICS (Golden Dataset Evaluation)")
        print(f"{'='*60}")
        print(f"Recall@{k}:    {retrieval_metrics.get('recall_at_k', 0):.2%}")
        print(f"Precision@{k}: {retrieval_metrics.get('precision_at_k', 0):.2%}")
        print(f"F1@{k}:        {retrieval_metrics.get('f1_at_k', 0):.2%}")
        print(f"Hit Rate:    {retrieval_metrics.get('hit_rate', 0):.2%}")
        print(f"MRR:         {retrieval_metrics.get('mrr', 0):.4f}")
        if "ndcg_at_k" in retrieval_metrics:
            print(f"nDCG@{k}:      {retrieval_metrics['ndcg_at_k']:.4f}")
        print(f"{'='*60}\n")

    return {
        "retrieved_docs": [docs_text],
        "retrieval_quality_score": quality_score,
        "retrieval_quality_reasoning": quality_reasoning,
        "retrieval_quality_issues": quality_issues,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "unique_docs_list": unique_docs,
        "retrieval_metrics": retrieval_metrics,
        "messages": [AIMessage(content=f"Retrieved {len(unique_docs)} documents")],
    }

# ============ REWRITING FOR INSUFFICIENT RESULTS ============

def rewrite_and_refine_node(state: dict) -> dict:
    """
    Rewrite query if retrieval quality is poor.

    Uses specific feedback from retrieval_quality_issues to guide rewriting.
    """

    query = state["active_query"]
    quality = state.get("retrieval_quality_score", 0)
    issues = state.get("retrieval_quality_issues", [])

    if issues:
        # Build actionable feedback based on specific issues detected
        feedback_parts = [
            f"Previous retrieval quality: {quality:.0%}",
            "",
            "Detected issues and recommended improvements:"
        ]

        for issue in issues:
            if issue == "partial_coverage":
                feedback_parts.append("- PARTIAL COVERAGE: Query aspects not fully addressed in retrieved documents.")
                feedback_parts.append("  Suggestion: Expand query to explicitly cover all aspects or break into sub-queries.")
            elif issue == "missing_key_info":
                feedback_parts.append("- MISSING KEY INFORMATION: Retrieved documents lack specific details needed to answer.")
                feedback_parts.append("  Suggestion: Add specific keywords, technical terms, or entities that might appear in relevant documents.")
            elif issue == "incomplete_context":
                feedback_parts.append("- INCOMPLETE CONTEXT: Documents provide insufficient depth or detail.")
                feedback_parts.append("  Suggestion: Add qualifiers or context to target more comprehensive sources (e.g., 'detailed explanation of', 'comprehensive guide to').")
            elif issue == "domain_misalignment":
                feedback_parts.append("- DOMAIN MISALIGNMENT: Retrieved documents are from wrong topic area or use different terminology.")
                feedback_parts.append("  Suggestion: Adjust terminology to match target domain (use domain-specific terms, acronyms, or jargon).")
            elif issue == "low_confidence" or issue == "insufficient_depth":
                feedback_parts.append("- LOW CONFIDENCE/DEPTH: Documents are surface-level or tangentially related.")
                feedback_parts.append("  Suggestion: Make query more specific and focused (add constraints, context, or narrow scope).")
            elif issue == "mixed_relevance":
                feedback_parts.append("- MIXED RELEVANCE: Some documents relevant, others off-topic.")
                feedback_parts.append("  Suggestion: Refine query to target more specific topic and reduce noise.")
            elif issue == "off_topic" or issue == "wrong_domain":
                feedback_parts.append("- OFF-TOPIC RESULTS: Documents retrieved are not relevant to query intent.")
                feedback_parts.append("  Suggestion: Rephrase query with different keywords or approach (try synonyms, related concepts, or reframe the question).")

        retrieval_context = "\n".join(feedback_parts)
    else:
        retrieval_context = f"Previous retrieval quality was {quality:.0%}. Improve query specificity and clarity."

    print(f"\n{'='*60}")
    print(f"QUERY REWRITING")
    print(f"Original query: {query}")
    print(f"Retrieval quality: {quality:.0%}")
    print(f"Issues detected: {', '.join(issues) if issues else 'None'}")
    print(f"{'='*60}\n")

    rewritten = rewrite_query(query, retrieval_context=retrieval_context)
    print(f"Rewritten query: {rewritten}")
    print(f"Note: Query expansions cleared - will regenerate for rewritten query")
    print(f"\nState clearing (semantic rewrite takes precedence):")
    print(f"  query_expansions: [] (will regenerate)")
    print(f"  retrieval_query: None (cleared to prevent stale optimization)")
    print(f"  active_query: {rewritten} (semantic rewrite)\n")

    return {
        "active_query": rewritten,
        "query_expansions": [],
        "retrieval_query": None,  # Clear stale algorithm optimization (semantic rewrite takes precedence)
        "messages": [AIMessage(content=f"Query rewritten: {query} → {rewritten}")],
    }

# ============ ANSWER GENERATION & EVALUATION ============

def answer_generation_node(state: dict) -> dict:
    """
    Generate answer using structured RAG prompt with unified retry handling.

    Implements RAG best practices: quality-aware thresholds, XML markup, unified feedback.
    Handles both initial generation and retries from combined evaluation.
    """

    question = state["baseline_query"]
    context = state["retrieved_docs"][-1] if state.get("retrieved_docs") else "No context"
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_retry = state.get("generation_retry_count", 0)
    retry_feedback = state.get("retry_feedback", "")

    print(f"\n{'='*60}")
    print(f"ANSWER GENERATION")
    print(f"Question: {question}")
    print(f"Context size: {len(context)} chars")
    print(f"Retrieval quality: {retrieval_quality:.0%}")
    print(f"Generation attempt: {generation_retry + 1}/3")
    print(f"{'='*60}\n")

    if not context or context == "No context":
        return {
            "final_answer": "I apologize, but I could not retrieve any relevant documents to answer your question. Please try rephrasing your query or check if the information exists in the knowledge base.",
            "messages": [AIMessage(content="Empty retrieval - no answer generated")],
        }

    formatted_context = context

    # Determine quality instruction based on retry scenario
    if generation_retry > 0 and retry_feedback:
        # Unified retry with combined feedback from evaluation
        quality_instruction = f"""RETRY GENERATION (Attempt {generation_retry + 1}/3)

Previous attempt had issues:
{retry_feedback}

Generate improved answer addressing ALL issues above while using the same retrieved context."""

        print(f"RETRY MODE:")
        print(f"Feedback:\n{retry_feedback}\n")
    else:
        # First generation - use quality-aware instructions
        if retrieval_quality > 0.8:
            quality_instruction = f"""High Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents are highly relevant and should contain the information needed to answer the question. Answer directly and confidently based on them."""
        elif retrieval_quality > 0.6:
            quality_instruction = f"""Medium Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents are somewhat relevant but may have gaps in coverage. Use them to answer what you can, but explicitly acknowledge any limitations or missing information."""
        else:
            quality_instruction = f"""Low Confidence Retrieval (Score: {retrieval_quality:.0%})
The retrieved documents may not fully address the question. Only answer what can be directly supported by the context. If the context is insufficient, clearly state: "The provided context does not contain enough information to answer this question completely." """

    spec = get_model_for_task("answer_generation")
    is_gpt5 = spec.name.lower().startswith("gpt-5")

    # For unified retry, hallucination feedback is already in retry_feedback
    hallucination_feedback = retry_feedback if (generation_retry > 0 and retry_feedback) else ""
    is_retry_after_hallucination = "HALLUCINATION DETECTED" in retry_feedback if retry_feedback else False

    system_prompt, user_message = get_answer_generation_prompts(
        hallucination_feedback=hallucination_feedback,
        quality_instruction=quality_instruction,
        formatted_context=formatted_context,
        question=question,
        is_gpt5=is_gpt5,
        is_retry_after_hallucination=is_retry_after_hallucination,
        unsupported_claims=None  # Claims already in retry_feedback
    )

    llm = _get_answer_generation_llm()
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])

    result = {
        "final_answer": response.content,
        "generation_retry_count": generation_retry + 1,
        "messages": [response],
    }

    return result

def get_quality_fix_guidance(issues: list[str]) -> str:
    """Generate fix guidance based on quality issues."""
    guidance_map = {
        "incomplete_synthesis": "Extract and synthesize key points from ALL documents",
        "lacks_specificity": "Include specific details (numbers, dates, names, technical terms)",
        "unsupported_claims": "Remove claims not explicitly stated in context",
        "wrong_focus": "Re-read question and address the primary intent",
        "partial_answer": "Ensure all question parts are answered completely",
        "missing_details": "Add sufficient depth and explanation",
        "contextual_gaps": "Provide necessary background context",
        "retrieval_limited": "Work with available information, state limitations if needed",
    }
    return "; ".join([guidance_map.get(issue, issue) for issue in issues])


def evaluate_answer_node(state: dict) -> dict:
    """
    Combined groundedness + quality evaluation (single decision point).

    Performs both checks in sequence:
    1. NLI-based hallucination detection (factuality)
    2. LLM-as-judge quality assessment (sufficiency)

    Returns unified decision: is answer good enough to return?
    """
    answer = state.get("final_answer", "")
    context = state.get("retrieved_docs", [""])[-1]
    question = state["baseline_query"]
    retrieval_quality = state.get("retrieval_quality_score", 0.7)
    generation_retry = state.get("generation_retry_count", 0)

    print(f"\n{'='*60}")
    print(f"ANSWER EVALUATION (Combined Groundedness + Quality)")
    print(f"Generation attempt: {generation_retry + 1}")
    print(f"Retrieval quality: {retrieval_quality:.0%}")

    # ==== 1. GROUNDEDNESS CHECK (NLI) ====

    # Handle proper refusal when retrieval poor
    has_hallucination = False
    groundedness_score = 1.0
    unsupported_claims = []

    if retrieval_quality < 0.6:
        refusal_patterns = [
            "context does not contain enough information",
            "provided context is insufficient",
            "cannot answer based on the context",
        ]
        if any(p in answer.lower() for p in refusal_patterns):
            print(f"Groundedness: PASS (proper refusal)")
            groundedness_score = 1.0
        else:
            # Poor retrieval but LLM tried to answer -> check groundedness
            groundedness_result = nli_detector.verify_groundedness(answer, context)
            groundedness_score = groundedness_result.get("groundedness_score", 1.0)
            has_hallucination = groundedness_score < 0.8
            unsupported_claims = groundedness_result.get("unsupported_claims", [])
            print(f"Groundedness: {groundedness_score:.0%} (poor retrieval, checked anyway)")
    else:
        # Normal groundedness check
        groundedness_result = nli_detector.verify_groundedness(answer, context)
        groundedness_score = groundedness_result.get("groundedness_score", 1.0)
        has_hallucination = groundedness_score < 0.8
        unsupported_claims = groundedness_result.get("unsupported_claims", [])
        print(f"Groundedness: {groundedness_score:.0%}")

    # ==== 2. QUALITY CHECK (LLM-as-judge) ====

    retrieval_quality_issues = state.get("retrieval_quality_issues", [])
    has_missing_info = any(issue in retrieval_quality_issues for issue in ["partial_coverage", "missing_key_info", "incomplete_context"])
    quality_threshold = 0.5 if (retrieval_quality < 0.6 or has_missing_info) else 0.65

    spec = get_model_for_task("answer_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
    structured_answer_llm = quality_llm.with_structured_output(AnswerQualityEvaluation)

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
        evaluation = structured_answer_llm.invoke(evaluation_prompt)
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

    # ==== 3. COMBINED DECISION ====

    has_issues = has_hallucination or not is_quality_sufficient

    # Build unified feedback for retry
    retry_feedback_parts = []
    if has_hallucination:
        retry_feedback_parts.append(
            f"HALLUCINATION DETECTED ({groundedness_score:.0%} grounded):\n"
            f"Unsupported claims: {', '.join(unsupported_claims)}\n"
            f"Fix: ONLY state facts explicitly in retrieved context."
        )
    if not is_quality_sufficient:
        retry_feedback_parts.append(
            f"QUALITY ISSUES:\n"
            f"Problems: {', '.join(quality_issues)}\n"
            f"Fix: {get_quality_fix_guidance(quality_issues)}"
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
        "messages": [AIMessage(content=f"Evaluation: {groundedness_score:.0%} grounded, {confidence:.0%} quality")],
    }
