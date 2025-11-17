from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from advanced_agentic_rag_langgraph.retrieval import (
    expand_query,
    rewrite_query,
    AdaptiveRetriever,
    LLMMetadataReRanker,
    SemanticRetriever,
)
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats
from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_ndcg
from advanced_agentic_rag_langgraph.validation import NLIHallucinationDetector
import re
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
adaptive_retriever = None  # Will be initialized via setup_retriever()
conversational_rewriter = ConversationalRewriter()  # For query rewriting
strategy_selector = StrategySelector()  # For intelligent strategy selection
nli_detector = NLIHallucinationDetector()  # NLI-based hallucination detection


# ============ STRUCTURED OUTPUT SCHEMAS ============

class RetrievalQualityEvaluation(TypedDict):
    """Structured output schema for retrieval quality assessment.

    Used with .with_structured_output() to ensure reliable parsing of LLM evaluations.
    Eliminates regex-based score extraction errors.
    """
    quality_score: float  # 0-100 scale
    reasoning: str  # Explanation of the quality score
    issues: list[str]  # Specific problems identified (empty list if none)


# ============ CONVERSATIONAL QUERY REWRITING ============

def conversational_rewrite_node(state: dict) -> dict:
    """
    Rewrite query using conversation history to make it self-contained.

    This node runs before query expansion to ensure queries have proper context.
    """
    question = state.get("question", state.get("original_query", ""))
    conversation_history = state.get("conversation_history", [])

    # Rewrite query if conversation history exists
    rewritten_query, reasoning = conversational_rewriter.rewrite(
        question,
        conversation_history
    )

    # Log rewrite if it happened
    if rewritten_query != question:
        print(f"\n{'='*60}")
        print(f"CONVERSATIONAL REWRITE")
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten_query}")
        print(f"Reasoning: {reasoning}")
        print(f"{'='*60}\n")

    return {
        "original_query": rewritten_query,  # Use rewritten as the "original" for rest of pipeline
        "question": question,  # Preserve raw user input
        "corpus_stats": get_corpus_stats(),  # Add corpus stats for strategy selection
        "messages": [HumanMessage(content=question)],
    }

# ============ QUERY OPTIMIZATION STAGE ============

def _should_skip_expansion_llm(query: str) -> bool:
    """
    Use LLM to determine if query expansion would improve retrieval.

    Domain-agnostic - works for any query type and corpus.
    More accurate than heuristics - handles context and intent.
    """
    expansion_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

Return ONLY 'yes' to expand or 'no' to skip expansion.
No explanation needed."""

    try:
        response = expansion_llm.invoke(prompt)
        decision = response.content.strip().lower()

        # Parse yes/no (handle variations)
        skip = decision.startswith('no')

        return skip

    except Exception as e:
        print(f"Warning: Expansion decision LLM failed: {e}, defaulting to expand")
        return False  # On error, expand (safer default)


def query_expansion_node(state: dict) -> dict:
    """
    Conditionally expand queries using LLM to assess expansion benefit.

    Aligns with best practices: 'Apply selective expansion using confidence thresholds'
    Domain-agnostic - works for any query type.
    """
    query = state["original_query"]

    # Use LLM to decide if expansion is beneficial
    if _should_skip_expansion_llm(query):
        print(f"\n{'='*60}")
        print(f"EXPANSION SKIPPED")
        print(f"Query: {query}")
        print(f"Reason: LLM determined query is clear/specific enough")
        print(f"{'='*60}\n")

        return {
            "query_expansions": [query],
            "current_query": query,
        }

    # Expand queries that would benefit from variation
    expansions = expand_query(query)
    print(f"\n{'='*60}")
    print(f"QUERY EXPANDED")
    print(f"Original: {query}")
    print(f"Expansions: {expansions[1:]}")
    print(f"{'='*60}\n")

    return {
        "query_expansions": expansions,
        "current_query": query,
    }

def decide_retrieval_strategy_node(state: dict) -> dict:
    """
    Decide which retrieval strategy to use based on query and corpus characteristics.

    Uses pure LLM classification for intelligent, domain-agnostic strategy selection.
    """
    query = state["current_query"]
    corpus_stats = state.get("corpus_stats", {})

    # Use intelligent strategy selector
    strategy, confidence, reasoning = strategy_selector.select_strategy(
        query,
        corpus_stats
    )

    # Log decision
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

# ============ HYBRID RETRIEVAL STAGE ============

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

    # ============ RRF-BASED MULTI-QUERY RETRIEVAL ============
    # Step 1: Retrieve for each query variant WITHOUT reranking (for RRF fusion)
    doc_ranks = {}  # {doc_id: [rank1, rank2, ...]}
    doc_objects = {}  # {doc_id: Document}

    for query in state["query_expansions"]:
        # Use retrieve_without_reranking() to get larger candidate pool for RRF
        docs = adaptive_retriever.retrieve_without_reranking(query, strategy=strategy)

        # Track rank position for each document in this query's results
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("id", doc.page_content[:50])

            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
                doc_objects[doc_id] = doc

            doc_ranks[doc_id].append(rank)

    # Step 2: Calculate RRF scores
    # Formula: score(doc) = sum(1/(rank + k)) across all queries
    # k=60 is standard constant from research
    k = 60
    rrf_scores = {}

    for doc_id, ranks in doc_ranks.items():
        rrf_score = sum(1.0 / (rank + k) for rank in ranks)
        rrf_scores[doc_id] = rrf_score

    # Step 3: Sort documents by RRF score (descending)
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Step 4: Reconstruct unique_docs in RRF-ranked order
    unique_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids]

    # Log RRF statistics for debugging/monitoring
    print(f"\n{'='*60}")
    print(f"RRF MULTI-QUERY RETRIEVAL")
    print(f"Query variants: {len(state['query_expansions'])}")
    print(f"Total retrievals: {sum(len(ranks) for ranks in doc_ranks.values())}")
    print(f"Unique docs after RRF: {len(unique_docs)}")
    if sorted_doc_ids[:3]:
        print(f"Top 3 RRF scores: {[f'{rrf_scores[doc_id]:.4f}' for doc_id in sorted_doc_ids[:3]]}")
    print(f"{'='*60}\n")

    # ============ TWO-STAGE RERANKING AFTER RRF ============
    # Step 5: Apply two-stage reranking to RRF-fused results
    # Limit input to top-40 for efficiency (prevents excessive CrossEncoder processing)
    reranking_input = unique_docs[:40]

    print(f"{'='*60}")
    print(f"TWO-STAGE RERANKING (After RRF)")
    print(f"Input: {len(reranking_input)} docs (from RRF top-40)")

    # Apply TwoStageReRanker: CrossEncoder (top-15) then LLM-as-judge (top-4)
    # Use rewritten_query if available, otherwise fall back to original_query
    query_for_reranking = state.get('rewritten_query', state.get('original_query', ''))
    ranked_results = adaptive_retriever.reranker.rank(
        query_for_reranking,
        reranking_input
    )

    # Extract documents and scores
    unique_docs = [doc for doc, score in ranked_results]
    reranking_scores = [score for doc, score in ranked_results]

    print(f"Output: {len(unique_docs)} docs after two-stage reranking")
    print(f"Reranking scores (top-3): {[f'{score:.4f}' for score in reranking_scores[:3]]}")
    print(f"{'='*60}\n")

    # Format for state
    docs_text = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in unique_docs[:5]  # Top 5
    ])

    # Evaluate retrieval quality using structured output
    quality_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_quality_llm = quality_llm.with_structured_output(RetrievalQualityEvaluation)

    quality_prompt = f"""Query: {state['original_query']}

Retrieved documents (top 5 after reranking):
{docs_text}

Evaluate retrieval quality to determine if these documents are sufficient for answer generation.

EVALUATION CRITERIA:

1. Coverage: How many aspects of the query are addressed?
   - Multi-aspect query (e.g., "advantages AND disadvantages"): Both aspects needed
   - Single-aspect query: Core information must be present
   - Consider: Are all parts of the question answered by the documents?

2. Completeness: Can the query be fully answered with these documents?
   - Complete information present: Documents contain everything needed
   - Partial information: Some details present but gaps exist
   - Insufficient: Cannot answer without additional sources

3. Relevance: Are documents on-topic and directly useful?
   - High relevance: Documents directly address query topic
   - Mixed relevance: Some docs relevant, others tangential
   - Low relevance: Documents off-topic or only peripherally related

SCORING GUIDELINES (0-100 scale, aligned with routing threshold of 60):

- 80-100: EXCELLENT - Proceed to answer generation immediately
  * All/most query aspects directly addressed
  * Complete information for full answer
  * All documents highly relevant to query

- 60-79: GOOD - Acceptable for answer generation [THRESHOLD: Will proceed]
  * Key query aspects covered (may have minor gaps)
  * Sufficient information for complete answer
  * Most documents relevant, minimal noise

- 40-59: FAIR - Requires query rewriting [THRESHOLD: Will retry if attempts < 2]
  * Partial coverage, key information missing
  * Incomplete information, gaps in answer
  * Documents tangential or only partially relevant

- 0-39: POOR - Inadequate retrieval, needs strategy change
  * Wrong domain or off-topic documents
  * Cannot answer query with current results
  * Most/all documents irrelevant

STRUCTURED OUTPUT:

- quality_score (0-100): Aggregate score following guidelines above

- reasoning: 2-3 sentences explaining:
  * Which aspects are covered vs missing
  * Whether information is complete for answering
  * Relevance quality of documents

- issues: List specific problems (empty list if none):
  * "missing_key_info": Required information not in documents (specify what is missing)
  * "partial_coverage": Some query aspects covered, others missing (list missing aspects)
  * "incomplete_context": Context lacks necessary details to fully answer query
  * "wrong_domain": Documents from unrelated topic area
  * "insufficient_depth": Surface-level info only, lacks detail
  * "off_topic": Documents irrelevant to query
  * "mixed_relevance": Combination of relevant and irrelevant docs

IMPORTANT: If key information or query aspects are missing, explicitly include "partial_coverage"
or "missing_key_info" in the issues list. This assessment is critical for routing decisions.

Return your evaluation as structured data."""

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

    # Calculate retrieval metrics if ground truth available (from golden dataset evaluation)
    retrieval_metrics = {}
    ground_truth_doc_ids = state.get("ground_truth_doc_ids")
    relevance_grades = state.get("relevance_grades")

    if ground_truth_doc_ids:
        # Binary relevance metrics (Recall, Precision, F1, Hit Rate, MRR)
        retrieval_metrics = calculate_retrieval_metrics(
            unique_docs,
            ground_truth_doc_ids,
            k=5
        )

        # Graded relevance metric (nDCG) if relevance grades available
        if relevance_grades:
            retrieval_metrics["ndcg_at_5"] = calculate_ndcg(
                unique_docs,
                relevance_grades,
                k=5
            )

        # Log metrics for debugging
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
        "retrieval_quality_reasoning": quality_reasoning,  # LLM explanation of quality score
        "retrieval_quality_issues": quality_issues,  # Specific problems identified
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "unique_docs_list": unique_docs,  # Store Document objects for metadata analysis
        "retrieval_metrics": retrieval_metrics,  # Store metrics for golden dataset evaluation
        "messages": [AIMessage(content=f"Retrieved {len(unique_docs)} documents")],
    }

# ============ REWRITING FOR INSUFFICIENT RESULTS ============

def rewrite_and_refine_node(state: dict) -> dict:
    """
    Rewrite query if retrieval quality is poor.

    Uses specific feedback from retrieval_quality_issues to guide rewriting,
    providing actionable instructions to the LLM for better query formulation.
    """

    query = state["current_query"]
    quality = state.get("retrieval_quality_score", 0)

    # Only rewrite if quality is poor AND we haven't done it too many times
    if quality < 0.6 and state.get("retrieval_attempts", 0) < 2:
        # Build specific feedback from retrieval issues
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
            # Fallback to generic feedback if no specific issues detected
            retrieval_context = f"Previous retrieval quality was {quality:.0%}. Improve query specificity and clarity."

        print(f"\n{'='*60}")
        print(f"QUERY REWRITING")
        print(f"Original query: {query}")
        print(f"Retrieval quality: {quality:.0%}")
        print(f"Issues detected: {', '.join(issues) if issues else 'None'}")
        print(f"{'='*60}\n")

        rewritten = rewrite_query(query, retrieval_context=retrieval_context)
        print(f"Rewritten query: {rewritten}\n")

        return {
            "current_query": rewritten,
            "rewritten_query": rewritten,
            "messages": [AIMessage(content=f"Query rewritten for better retrieval")],
        }
    else:
        return {
            "rewritten_query": query,
        }

# ============ ANSWER GENERATION & EVALUATION ============

def answer_generation_with_quality_node(state: dict) -> dict:
    """
    Generate answer using structured RAG prompt with metadata enrichment.

    Domain-agnostic design: Works with any document type (research papers,
    tutorials, contracts, manuals, blog posts, documentation, etc.)

    Implements RAG prompting best practices:
    - Numbered documents with metadata (type, level, domain)
    - Quality-aware instructions with explicit thresholds
    - Structured template with XML-like section markers
    - Optional citations (model decides when helpful)
    - Clear insufficient-context handling
    - Groundedness feedback for hallucination self-correction
    """

    question = state["original_query"]
    context = state["retrieved_docs"][-1] if state.get("retrieved_docs") else "No context"
    quality_score = state.get("retrieval_quality_score", 0)

    # Check if this is a groundedness retry (feedback mechanism)
    retry_needed = state.get("retry_needed", False)
    unsupported_claims = state.get("unsupported_claims", [])
    groundedness_score = state.get("groundedness_score", 1.0)

    # Handle empty retrieval gracefully
    if not context or context == "No context":
        return {
            "final_answer": "I apologize, but I could not retrieve any relevant documents to answer your question. Please try rephrasing your query or check if the information exists in the knowledge base.",
            "messages": [AIMessage(content="Empty retrieval - no answer generated")],
        }

    # Context is already formatted as string from retrieve_with_expansion_node
    # Format: "[source] content\n---\n[source] content..."
    # We'll enhance it with document numbering and structure
    formatted_context = context  # Use existing formatted context as-is for now

    # Build hallucination feedback if this is a retry after groundedness check
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

    # Quality-aware instructions with explicit thresholds
    if quality_score > 0.8:
        quality_instruction = f"""High Confidence Retrieval (Score: {quality_score:.0%})
The retrieved documents are highly relevant and should contain the information needed to answer the question. Answer directly and confidently based on them."""
    elif quality_score > 0.6:
        quality_instruction = f"""Medium Confidence Retrieval (Score: {quality_score:.0%})
The retrieved documents are somewhat relevant but may have gaps in coverage. Use them to answer what you can, but explicitly acknowledge any limitations or missing information."""
    else:
        quality_instruction = f"""Low Confidence Retrieval (Score: {quality_score:.0%})
The retrieved documents may not fully address the question. Only answer what can be directly supported by the context. If the context is insufficient, clearly state: "The provided context does not contain enough information to answer this question completely." """

    # Domain-agnostic system prompt with optional citations
    # Prepend hallucination feedback if this is a retry
    system_prompt = f"""{hallucination_feedback}You are an AI assistant that answers questions based exclusively on retrieved documents. Your role is to provide accurate, well-grounded responses using only the information present in the provided context.

{quality_instruction}

Core Instructions:
1. Base your answer ONLY on the provided context - do not use external knowledge or make assumptions beyond what is explicitly stated
2. If the context does not contain sufficient information to answer the question, clearly state: "The provided context does not contain enough information to answer this question."
3. Provide direct, concise answers that extract and synthesize the relevant information
4. When helpful for clarity or verification, you may reference specific documents (e.g., "Document 2 explains that..." or "According to the retrieved information...")
5. Match your confidence level to the retrieval quality - acknowledge uncertainty when present"""

    # Structured user message with XML-like markup
    user_message = f"""<retrieved_context>
{formatted_context}
</retrieved_context>

<question>
{question}
</question>

<instructions>
1. Answer the question using ONLY information from the <retrieved_context> section above
2. If the context is insufficient, respond: "The provided context does not contain enough information to answer this question."
3. Provide a direct, accurate answer that synthesizes the relevant information
4. If multiple documents contain relevant information, combine insights appropriately
5. Do not make assumptions or inferences beyond what is explicitly stated
</instructions>

<answer>
"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])

    return {
        "final_answer": response.content,
        "messages": [response],
    }

def groundedness_check_node(state: dict) -> dict:
    """
    Verify that generated answer is factually grounded in retrieved context.

    Implements RAG Triad framework - Groundedness dimension.
    Uses NLI-based claim verification for improved accuracy (0.83 F1 vs 0.70 F1).

    NLI Approach:
    1. Decompose answer into atomic claims (LLM)
    2. Verify each claim against context (NLI model)
    3. Calculate groundedness = supported claims / total claims

    Conditional blocking strategy (best practice):
    - Score < 0.6: Severe hallucination, flag for retry
    - Score 0.6-0.8: Moderate hallucination, flag with warning
    - Score >= 0.8: Good groundedness, proceed normally
    """
    answer = state.get("final_answer", "")
    context = state.get("retrieved_docs", [""])[-1]  # Most recent retrieved context

    # Use NLI-based hallucination detector
    # Two-step: claim decomposition + entailment verification
    evaluation = nli_detector.verify_groundedness(answer, context)

    groundedness_score = evaluation.get("groundedness_score", 1.0)
    unsupported_claims = evaluation.get("unsupported_claims", [])
    claims = evaluation.get("claims", [])

    # Conditional hallucination detection
    if groundedness_score < 0.6:
        # SEVERE: High hallucination risk
        has_hallucination = True
        retry_needed = True
        severity = "SEVERE"
    elif groundedness_score < 0.8:
        # MODERATE: Some unsupported claims
        has_hallucination = True
        retry_needed = False
        severity = "MODERATE"
    else:
        # GOOD: Most claims supported
        has_hallucination = False
        retry_needed = False
        severity = "NONE"

    # Log hallucination detection
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

    # Increment retry counter if we're retrying
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

    Streamlined: Context completeness now assessed in retrieval_quality_score
    (via "partial_coverage" and "missing_key_info" issues), eliminating redundant LLM call.
    """
    question = state["original_query"]
    answer = state.get("final_answer", "")
    retrieval_quality = state.get("retrieval_quality_score", 0)
    retrieval_quality_issues = state.get("retrieval_quality_issues", [])

    # ============ ANSWER QUALITY EVALUATION ============
    # Lower the quality threshold if retrieval indicated missing information
    has_missing_info = any(issue in retrieval_quality_issues for issue in ["partial_coverage", "missing_key_info", "incomplete_context"])
    quality_threshold = 0.5 if (retrieval_quality < 0.6 or has_missing_info) else 0.65

    evaluation_prompt = f"""Question: {question}
Answer: {answer}
Retrieval quality: {retrieval_quality:.0%}
Detected issues: {', '.join(retrieval_quality_issues) if retrieval_quality_issues else 'None'}

Evaluate this answer:
1. Relevant? (yes/no)
2. Complete? (yes/no)
3. Accurate? (yes/no)
4. Confidence (0-100)

Note: Retrieval quality and detected issues should inform your confidence assessment.

Response as JSON:
{{
    "is_relevant": boolean,
    "is_complete": boolean,
    "is_accurate": boolean,
    "confidence_score": number,
    "reasoning": "brief note"
}}"""

    response = llm.invoke(evaluation_prompt)

    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        evaluation = json.loads(json_match.group()) if json_match else {}
    except:
        evaluation = {"confidence_score": 70}

    confidence = evaluation.get("confidence_score", 70) / 100
    is_sufficient = (
        evaluation.get("is_relevant", True) and
        evaluation.get("is_complete", True) and
        confidence >= quality_threshold
    )

    return {
        # Answer evaluation fields (context completeness now in retrieval_quality_issues)
        "is_answer_sufficient": is_sufficient,
        "confidence_score": confidence,
        "messages": [AIMessage(content=f"Evaluation: Confidence={confidence:.0%}")],
    }
