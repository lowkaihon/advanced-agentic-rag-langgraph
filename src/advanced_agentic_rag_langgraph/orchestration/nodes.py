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


def _get_answer_generation_llm():
    """Get LLM for answer generation with tier-based configuration."""
    spec = get_model_for_task("answer_generation")
    return ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
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
    question = state.get("user_question", state.get("baseline_query", ""))

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
        print(f"{'='*60}\n")

    return {
        "baseline_query": rewritten_query,
        "user_question": question,
        "corpus_stats": get_corpus_stats(),
        "messages": [HumanMessage(content=question)],
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
        return skip
    except Exception as e:
        print(f"Warning: Expansion decision LLM failed: {e}, defaulting to expand")
        return False


def query_expansion_node(state: dict) -> dict:
    """
    Conditionally expand queries using LLM to assess expansion benefit.

    Also handles early strategy switch state updates (moved from route_after_retrieval
    to maintain router purity per LangGraph best practices).

    Three routing scenarios:
    1. Early switch: route_after_retrieval detected strategy mismatch
    2. Re-retrieval: route_after_evaluation triggered retrieval-caused hallucination fix
    3. Late switch: route_after_evaluation determined answer insufficient
    """

    quality = state.get("retrieval_quality_score", 1.0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])
    strategy_changed = state.get("strategy_changed", False)
    current_strategy = state.get("retrieval_strategy", "hybrid")
    retrieval_caused_hallucination = state.get("retrieval_caused_hallucination", False)
    is_answer_sufficient = state.get("is_answer_sufficient", True)

    early_switch = (quality <= 0.6 and
                    attempts < 2 and
                    not strategy_changed and
                    ("off_topic" in issues or "wrong_domain" in issues))

    if early_switch:
        # NOTE: Import inside function to avoid circular import (graph.py imports from nodes.py)
        from advanced_agentic_rag_langgraph.orchestration.graph import select_next_strategy

        next_strategy = select_next_strategy(current_strategy, issues)
        optimized_query = optimize_query_for_strategy(
            query=state.get("active_query", state["baseline_query"]),
            new_strategy=next_strategy,
            old_strategy=current_strategy,
            issues=issues
        )

        print(f"\n{'='*60}")
        print(f"EARLY STRATEGY SWITCH")
        print(f"From: {current_strategy} to {next_strategy}")
        print(f"Reason: {', '.join(issues)}")
        print(f"Attempt: {attempts + 1}")
        print(f"Quality score: {quality:.0%}")
        print(f"Note: Query optimized for {next_strategy} strategy")
        print(f"{'='*60}\n")

        query = optimized_query
        updates = {
            "active_query": optimized_query,
            "retrieval_strategy": next_strategy,
            "strategy_switch_reason": f"Early detection: {', '.join(issues)}",
            "strategy_changed": True,
            "query_expansions": [],
        }
        state.update(updates)

    elif retrieval_caused_hallucination and attempts < 3:
        if current_strategy == "semantic":
            next_strategy = "keyword"
        elif current_strategy == "keyword":
            next_strategy = "hybrid"
        else:
            next_strategy = "semantic"

        optimized_query = optimize_query_for_strategy(
            query=state.get("active_query", state["baseline_query"]),
            new_strategy=next_strategy,
            old_strategy=current_strategy,
            issues=["retrieval_caused_hallucination"]
        )

        print(f"\n{'='*60}")
        print(f"RE-RETRIEVAL (Hallucination Mitigation)")
        print(f"Trigger: Poor retrieval caused hallucination")
        print(f"Strategy: {current_strategy} to {next_strategy}")
        print(f"Attempt: {attempts + 1}/3")
        print(f"Research: Re-retrieval > regeneration for context gaps")
        print(f"Note: Query optimized for {next_strategy} strategy")
        print(f"{'='*60}\n")

        query = optimized_query
        updates = {
            "active_query": optimized_query,
            "retrieval_strategy": next_strategy,
            "strategy_changed": True,
            "query_expansions": [],
            "retrieval_caused_hallucination": False,
        }
        state.update(updates)

    elif not is_answer_sufficient and attempts < 3 and not retrieval_caused_hallucination:
        retrieval_quality_issues = state.get("retrieval_quality_issues", [])
        retrieval_quality_score = state.get("retrieval_quality_score", 0.7)

        if "missing_key_info" in retrieval_quality_issues and retrieval_quality_score < 0.6:
            if current_strategy != "semantic":
                next_strategy = "semantic"
                reasoning = f"Content-driven: Missing key information detected, switching to semantic search for better conceptual coverage"
            else:
                next_strategy = "hybrid"
                reasoning = f"Content-driven: Semantic failed to find key information, trying hybrid for broader coverage"
        elif "off_topic" in retrieval_quality_issues or "wrong_domain" in retrieval_quality_issues:
            if current_strategy != "keyword":
                next_strategy = "keyword"
                reasoning = f"Content-driven: Off-topic results detected, switching to keyword search for precision"
            else:
                next_strategy = "hybrid"
                reasoning = f"Content-driven: Keyword search not precise enough, trying hybrid"
        elif "partial_coverage" in retrieval_quality_issues or "incomplete_context" in retrieval_quality_issues:
            if current_strategy == "hybrid":
                next_strategy = "semantic"
                reasoning = "Content-driven: Partial coverage with hybrid, trying semantic for depth"
            elif current_strategy == "semantic":
                next_strategy = "keyword"
                reasoning = "Content-driven: Semantic incomplete, trying keyword for specificity"
            else:
                next_strategy = "hybrid"
                reasoning = "Content-driven: Keyword insufficient, trying hybrid for balance"
        else:
            if current_strategy == "hybrid":
                next_strategy = "semantic"
                reasoning = "Fallback: hybrid to semantic"
            elif current_strategy == "semantic":
                next_strategy = "keyword"
                reasoning = "Fallback: semantic to keyword"
            else:
                next_strategy = current_strategy
                reasoning = "No strategy change (exhausted options)"

        refinement = {
            "iteration": attempts,
            "from_strategy": current_strategy,
            "to_strategy": next_strategy,
            "reasoning": reasoning,
            "retrieval_quality_issues": retrieval_quality_issues,
            "retrieval_quality_score": retrieval_quality_score,
        }

        strategy_changed_flag = (next_strategy != current_strategy)
        optimized_query = state.get("active_query", state["baseline_query"])

        if strategy_changed_flag:
            answer_quality_issues = state.get("answer_quality_issues", [])
            combined_issues = list(set(retrieval_quality_issues + answer_quality_issues))
            optimized_query = optimize_query_for_strategy(
                query=state.get("active_query", state["baseline_query"]),
                new_strategy=next_strategy,
                old_strategy=current_strategy,
                issues=combined_issues
            )

        print(f"\n{'='*60}")
        print(f"STRATEGY REFINEMENT")
        print(f"Iteration: {refinement['iteration']}")
        print(f"Switch: {current_strategy} to {next_strategy}")
        print(f"Reasoning: {reasoning}")
        print(f"Retrieval quality: {retrieval_quality_score:.0%}")
        print(f"Detected issues: {', '.join(retrieval_quality_issues) if retrieval_quality_issues else 'None'}")
        if strategy_changed_flag:
            print(f"Strategy changed: Query optimized and will regenerate expansions")
        print(f"{'='*60}\n")

        query = optimized_query
        updates = {
            "active_query": optimized_query,
            "retrieval_strategy": next_strategy,
            "refinement_history": [refinement],
            "query_expansions": [],
            "strategy_changed": True,
        }
        state.update(updates)

    query = state.get("active_query", state["baseline_query"])
    result = {}

    if early_switch:
        result.update({
            "active_query": state["active_query"],
            "retrieval_strategy": state["retrieval_strategy"],
            "strategy_switch_reason": state.get("strategy_switch_reason", ""),
            "strategy_changed": state["strategy_changed"],
        })
    elif retrieval_caused_hallucination and attempts < 3:
        result.update({
            "active_query": state["active_query"],
            "retrieval_strategy": state["retrieval_strategy"],
            "strategy_changed": state["strategy_changed"],
            "retrieval_caused_hallucination": state["retrieval_caused_hallucination"],
        })
    elif not is_answer_sufficient and attempts < 3 and not retrieval_caused_hallucination:
        result.update({
            "active_query": state["active_query"],
            "retrieval_strategy": state["retrieval_strategy"],
            "refinement_history": [state["refinement_history"][-1]] if state.get("refinement_history") else [],
            "strategy_changed": state["strategy_changed"],
        })

    if _should_skip_expansion_llm(query):
        print(f"\n{'='*60}")
        print(f"EXPANSION SKIPPED")
        print(f"Query: {query}")
        print(f"Reason: LLM determined query is clear/specific enough")
        print(f"{'='*60}\n")

        result.update({
            "query_expansions": [query],
            "active_query": query,
        })
        return result

    expansions = expand_query(query)
    print(f"\n{'='*60}")
    print(f"QUERY EXPANDED")
    print(f"Original: {query}")
    print(f"Expansions: {expansions[1:]}")
    print(f"{'='*60}\n")

    result.update({
        "query_expansions": expansions,
        "active_query": query,
    })
    return result

def decide_retrieval_strategy_node(state: dict) -> dict:
    """
    Decide which retrieval strategy to use based on query and corpus characteristics.

    Uses pure LLM classification for intelligent, domain-agnostic strategy selection.
    """
    query = state["active_query"]
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

    for query in state.get("query_expansions", []):
        docs = adaptive_retriever.retrieve_without_reranking(query, strategy=strategy)

        for rank, doc in enumerate(docs):
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

    query_for_reranking = state.get('active_query', state.get('baseline_query', ''))
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
        for doc in unique_docs[:5]
    ])

    spec = get_model_for_task("retrieval_quality_eval")
    quality_llm = ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )
    structured_quality_llm = quality_llm.with_structured_output(RetrievalQualityEvaluation)

    quality_prompt = get_prompt("retrieval_quality_eval", query=state['baseline_query'], docs_text=docs_text)

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
            k=5
        )

        if relevance_grades:
            retrieval_metrics["ndcg_at_5"] = calculate_ndcg(
                unique_docs,
                relevance_grades,
                k=5
            )

        print(f"\n{'='*60}")
        print(f"RETRIEVAL METRICS (Golden Dataset Evaluation)")
        print(f"{'='*60}")
        print(f"Recall@5:    {retrieval_metrics.get('recall_at_k', 0):.2%}")
        print(f"Precision@5: {retrieval_metrics.get('precision_at_k', 0):.2%}")
        print(f"F1@5:        {retrieval_metrics.get('f1_at_k', 0):.2%}")
        print(f"Hit Rate:    {retrieval_metrics.get('hit_rate', 0):.2%}")
        print(f"MRR:         {retrieval_metrics.get('mrr', 0):.4f}")
        if "ndcg_at_5" in retrieval_metrics:
            print(f"nDCG@5:      {retrieval_metrics['ndcg_at_5']:.4f}")
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
    print(f"Note: Query expansions cleared - will regenerate for rewritten query\n")

    return {
        "active_query": rewritten,
        "query_expansions": [],
        "messages": [AIMessage(content=f"Query rewritten: {query} → {rewritten}")],
    }

# ============ ANSWER GENERATION & EVALUATION ============

def answer_generation_with_quality_node(state: dict) -> dict:
    """
    Generate answer using structured RAG prompt with quality-aware instructions.

    Implements RAG best practices: quality-aware thresholds, XML markup, groundedness feedback.
    Handles retry counter increment (moved from router for purity).
    """

    question = state["baseline_query"]
    context = state["retrieved_docs"][-1] if state.get("retrieved_docs") else "No context"
    quality_score = state.get("retrieval_quality_score", 0)
    retry_needed = state.get("retry_needed", False)
    unsupported_claims = state.get("unsupported_claims", [])
    groundedness_score = state.get("groundedness_score", 1.0)

    retry_count = state.get("groundedness_retry_count", 0)
    retrieval_quality = state.get("retrieval_quality_score", 0.7)

    if retry_needed and retry_count < 2 and retrieval_quality >= 0.6 and groundedness_score < 0.6:
        retry_count = retry_count + 1

    if not context or context == "No context":
        return {
            "final_answer": "I apologize, but I could not retrieve any relevant documents to answer your question. Please try rephrasing your query or check if the information exists in the knowledge base.",
            "messages": [AIMessage(content="Empty retrieval - no answer generated")],
        }

    formatted_context = context

    hallucination_feedback = ""
    if retry_needed and unsupported_claims:
        hallucination_feedback = f"""CRITICAL - GROUNDEDNESS ISSUE DETECTED:
Your previous answer had a groundedness score of {groundedness_score:.0%}, indicating it contained claims that were NOT supported by the retrieved documents.

Unsupported claims from previous attempt:
{chr(10).join(f"  {i+1}. {claim}" for i, claim in enumerate(unsupported_claims))}

REGENERATION REQUIREMENTS:
1. Use ONLY information that is explicitly stated in the retrieved context below
2. For each of the unsupported claims listed above, either:
   - Find direct supporting evidence in the context and rephrase the claim accurately with that evidence
   - Completely omit the claim if no supporting evidence exists in the context
3. When helpful for verification, you may reference the documents (e.g., "According to the retrieved information..." or "The documents indicate that...")
4. Be conservative: If you cannot find explicit support for a claim, do not include it
5. If the context is insufficient to answer the question fully, clearly state: "The provided context does not contain enough information to answer this question completely."

Your goal is to be factually grounded, not comprehensive. Quality over completeness.

"""
        print(f"\n{'='*60}")
        print(f"GROUNDEDNESS FEEDBACK PROVIDED")
        print(f"Previous groundedness: {groundedness_score:.0%}")
        print(f"Unsupported claims: {len(unsupported_claims)}")
        print(f"Regenerating with hallucination-specific instructions")
        print(f"{'='*60}\n")

    if quality_score > 0.8:
        quality_instruction = f"""High Confidence Retrieval (Score: {quality_score:.0%})
The retrieved documents are highly relevant and should contain the information needed to answer the question. Answer directly and confidently based on them."""
    elif quality_score > 0.6:
        quality_instruction = f"""Medium Confidence Retrieval (Score: {quality_score:.0%})
The retrieved documents are somewhat relevant but may have gaps in coverage. Use them to answer what you can, but explicitly acknowledge any limitations or missing information."""
    else:
        quality_instruction = f"""Low Confidence Retrieval (Score: {quality_score:.0%})
The retrieved documents may not fully address the question. Only answer what can be directly supported by the context. If the context is insufficient, clearly state: "The provided context does not contain enough information to answer this question completely." """

    spec = get_model_for_task("answer_generation")
    is_gpt5 = spec.name.lower().startswith("gpt-5")

    # Detect if this is a retry after hallucination detection
    is_retry_after_hallucination = retry_needed and unsupported_claims and len(unsupported_claims) > 0

    system_prompt, user_message = get_answer_generation_prompts(
        hallucination_feedback=hallucination_feedback,
        quality_instruction=quality_instruction,
        formatted_context=formatted_context,
        question=question,
        is_gpt5=is_gpt5,
        is_retry_after_hallucination=is_retry_after_hallucination,
        unsupported_claims=unsupported_claims if is_retry_after_hallucination else None
    )

    llm = _get_answer_generation_llm()
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])

    result = {
        "final_answer": response.content,
        "messages": [response],
    }

    if retry_needed and retrieval_quality >= 0.6 and groundedness_score < 0.6:
        result["groundedness_retry_count"] = retry_count

    return result

def groundedness_check_node(state: dict) -> dict:
    """
    Verify answer is factually grounded in retrieved context.

    Uses NLI-based claim verification (0.83 F1 vs 0.70 F1).
    Three-tier severity: <0.6 severe (retry), 0.6-0.8 moderate (warn), >=0.8 good (proceed).
    """
    answer = state.get("final_answer", "")
    context = state.get("retrieved_docs", [""])[-1]

    evaluation = nli_detector.verify_groundedness(answer, context)

    groundedness_score = evaluation.get("groundedness_score", 1.0)
    unsupported_claims = evaluation.get("unsupported_claims", [])
    claims = evaluation.get("claims", [])

    if groundedness_score < 0.6:
        has_hallucination = True
        retry_needed = True
        severity = "SEVERE"
    elif groundedness_score < 0.8:
        has_hallucination = True
        retry_needed = False
        severity = "MODERATE"
    else:
        has_hallucination = False
        retry_needed = False
        severity = "NONE"

    if has_hallucination:
        print(f"\n{'='*60}")
        print(f"HALLUCINATION DETECTED - Severity: {severity}")
        print(f"Groundedness Score: {groundedness_score:.0%}")
        print(f"Total Claims: {len(claims)}")
        print(f"Unsupported Claims ({len(unsupported_claims)}):")
        for claim in unsupported_claims:
            print(f"  - {claim}")
        print(f"Action: {'RETRY GENERATION' if retry_needed else 'FLAG WARNING'}")
        print(f"{'='*60}\n")

    current_retry_count = state.get("groundedness_retry_count", 0)
    new_retry_count = current_retry_count + 1 if retry_needed else current_retry_count

    return {
        "groundedness_score": groundedness_score,
        "has_hallucination": has_hallucination,
        "unsupported_claims": unsupported_claims,
        "retry_needed": retry_needed,
        "groundedness_severity": severity,
        "groundedness_retry_count": new_retry_count,
        "messages": [AIMessage(content=f"Groundedness: {groundedness_score:.0%} ({severity})")],
    }


def evaluate_answer_with_retrieval_node(state: dict) -> dict:
    """
    Evaluate answer quality considering retrieval quality.

    Handles re-retrieval flag setting (moved from router for purity).
    Uses vRAG-Eval framework with adaptive thresholds.
    """
    question = state["baseline_query"]
    answer = state.get("final_answer", "")
    retrieval_quality = state.get("retrieval_quality_score", 0)
    retrieval_quality_issues = state.get("retrieval_quality_issues", [])
    retry_needed = state.get("retry_needed", False)
    retry_count = state.get("groundedness_retry_count", 0)
    groundedness_score = state.get("groundedness_score", 1.0)

    result_updates = {}

    if retry_needed and retry_count < 2 and retrieval_quality < 0.6 and groundedness_score < 0.6:
        result_updates["retrieval_caused_hallucination"] = True
        result_updates["is_answer_sufficient"] = False

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
        issues = evaluation["issues"]
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
        issues = evaluation["issues"]

    is_sufficient = (
        evaluation["is_relevant"] and
        evaluation["is_complete"] and
        evaluation["is_accurate"] and
        confidence >= quality_threshold
    )

    final_result = {
        "is_answer_sufficient": is_sufficient,
        "confidence_score": confidence,
        "answer_quality_reasoning": reasoning,
        "answer_quality_issues": issues,
        "messages": [AIMessage(content=f"Evaluation: Confidence={confidence:.0%}")],
    }

    final_result.update(result_updates)

    return final_result
