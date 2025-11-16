from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from advanced_agentic_rag_langgraph.retrieval import (
    expand_query,
    rewrite_query,
    HybridRetriever,
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
hybrid_retriever = None  # Will be initialized via setup_retriever()
conversational_rewriter = ConversationalRewriter()  # For query rewriting
strategy_selector = StrategySelector()  # For intelligent strategy selection
nli_detector = NLIHallucinationDetector()  # NLI-based hallucination detection

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
    """Retrieve documents using query expansions and selected strategy"""

    global hybrid_retriever
    if hybrid_retriever is None:
        hybrid_retriever = setup_retriever()

    strategy = state.get("retrieval_strategy", "hybrid")

    # Retrieve using all query variations
    all_docs = []
    for query in state["query_expansions"]:
        docs = hybrid_retriever.retrieve(query, strategy=strategy)
        all_docs.extend(docs)

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        if doc_id not in seen:
            unique_docs.append(doc)
            seen.add(doc_id)

    # Format for state
    docs_text = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in unique_docs[:5]  # Top 5
    ])

    # Evaluate retrieval quality
    quality_prompt = f"""Query: {state['original_query']}

Retrieved documents:
{docs_text}

Rate the quality of these results (0-100):
- 0-30: Poor - irrelevant or off-topic
- 31-60: Fair - some relevance but incomplete
- 61-85: Good - relevant and useful
- 86-100: Excellent - directly answers the query"""

    response = llm.invoke(quality_prompt)

    # Extract score
    score_match = re.search(r'\d+', response.content)
    quality_score = float(score_match.group()) / 100 if score_match else 0.5

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
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "unique_docs_list": unique_docs,  # Store Document objects for metadata analysis
        "retrieval_metrics": retrieval_metrics,  # Store metrics for golden dataset evaluation
        "messages": [AIMessage(content=f"Retrieved {len(unique_docs)} documents")],
    }

# ============ METADATA-DRIVEN ADAPTIVE RETRIEVAL ============

def analyze_retrieved_metadata_node(state: dict) -> dict:
    """
    Analyze metadata of retrieved documents to detect quality issues.

    Examines:
    - Strategy mismatch: Do retrieved docs prefer different strategy?
    - Technical level distribution: Complexity alignment
    - Domain alignment: Topic relevance
    - Confidence metrics: Strategy certainty
    """
    docs = state.get("unique_docs_list", [])
    current_strategy = state.get("retrieval_strategy", "hybrid")

    if not docs:
        return {
            "doc_metadata_analysis": {},
            "strategy_mismatch_rate": 0.0,
            "avg_doc_confidence": 0.5,
            "domain_alignment_score": 0.5,
        }

    # Analyze strategy preferences from retrieved documents
    strategy_preferences = {}
    confidence_scores = []
    technical_levels = {}
    domains = {}

    for doc in docs:
        meta = doc.metadata

        # Count strategy preferences
        preferred_strategy = meta.get("best_retrieval_strategy", "hybrid")
        strategy_preferences[preferred_strategy] = strategy_preferences.get(preferred_strategy, 0) + 1

        # Collect confidence scores
        confidence = meta.get("strategy_confidence", 0.5)
        confidence_scores.append(confidence)

        # Count technical levels
        level = meta.get("technical_level", "intermediate")
        technical_levels[level] = technical_levels.get(level, 0) + 1

        # Count domains
        domain = meta.get("domain", "general")
        domains[domain] = domains.get(domain, 0) + 1

    total_docs = len(docs)

    # Calculate strategy mismatch rate
    docs_preferring_current = strategy_preferences.get(current_strategy, 0)
    strategy_mismatch_rate = 1.0 - (docs_preferring_current / total_docs)

    # Calculate average confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

    # Determine dominant strategy from docs
    dominant_strategy = max(strategy_preferences, key=strategy_preferences.get) if strategy_preferences else current_strategy

    # Determine dominant domain
    dominant_domain = max(domains, key=domains.get) if domains else "general"

    # Determine dominant technical level
    dominant_level = max(technical_levels, key=technical_levels.get) if technical_levels else "intermediate"

    # Calculate domain alignment (simplified - could be enhanced with query analysis)
    # For now, high alignment if one domain dominates
    domain_alignment = max(domains.values()) / total_docs if domains else 0.5

    # Detect quality issues
    quality_issues = []

    # Issue 1: Strategy mismatch
    if strategy_mismatch_rate > 0.6:
        quality_issues.append({
            "issue": "strategy_mismatch",
            "severity": "high",
            "description": f"{strategy_mismatch_rate:.0%} of docs prefer {dominant_strategy}, not {current_strategy}",
            "suggested_strategy": dominant_strategy
        })

    # Issue 2: Low confidence
    if avg_confidence < 0.5:
        quality_issues.append({
            "issue": "low_confidence",
            "severity": "medium",
            "description": f"Average doc confidence is {avg_confidence:.0%}",
            "suggested_strategy": "hybrid"  # Fallback to hybrid when uncertain
        })

    # Issue 3: Mixed technical levels (potential complexity mismatch)
    if len(technical_levels) >= 3:  # All three levels present
        quality_issues.append({
            "issue": "mixed_complexity",
            "severity": "low",
            "description": f"Documents span all complexity levels: {technical_levels}",
            "suggested_action": "Adjust k-values to favor {dominant_level} documents"
        })

    # Issue 4: Low domain alignment (domain misalignment)
    if domain_alignment < 0.6:
        quality_issues.append({
            "issue": "domain_misalignment",
            "severity": "medium",
            "description": f"Documents span multiple domains (alignment: {domain_alignment:.0%}). Dominant: {dominant_domain}",
            "suggested_strategy": "semantic"  # Semantic search may better capture domain-specific concepts
        })

    # Build analysis summary
    analysis = {
        "total_docs": total_docs,
        "strategy_preferences": strategy_preferences,
        "dominant_strategy": dominant_strategy,
        "dominant_domain": dominant_domain,
        "dominant_technical_level": dominant_level,
        "technical_level_distribution": technical_levels,
        "domain_distribution": domains,
        "quality_issues": quality_issues,
    }

    # Log metadata analysis
    print(f"\n{'='*60}")
    print(f"METADATA ANALYSIS")
    print(f"Current strategy: {current_strategy}")
    print(f"Docs preferring current: {docs_preferring_current}/{total_docs}")
    print(f"Strategy mismatch rate: {strategy_mismatch_rate:.0%}")
    print(f"Dominant strategy: {dominant_strategy}")
    print(f"Avg confidence: {avg_confidence:.0%}")
    print(f"Quality issues: {len(quality_issues)}")
    for issue in quality_issues:
        print(f"  - {issue['issue']}: {issue['description']}")
    print(f"{'='*60}\n")

    return {
        "doc_metadata_analysis": analysis,
        "strategy_mismatch_rate": strategy_mismatch_rate,
        "avg_doc_confidence": avg_confidence,
        "domain_alignment_score": domain_alignment,
    }

# ============ REWRITING FOR INSUFFICIENT RESULTS ============

def rewrite_and_refine_node(state: dict) -> dict:
    """Rewrite query if retrieval quality is poor"""

    query = state["current_query"]
    quality = state.get("retrieval_quality_score", 0)

    # Only rewrite if quality is poor AND we haven't done it too many times
    if quality < 0.6 and state.get("retrieval_attempts", 0) < 2:
        rewritten = rewrite_query(query, retrieval_context="Insufficient results")
        print(f"Rewritten: {query} to {rewritten}")

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
    """Generate answer and include retrieval quality in reasoning"""

    question = state["original_query"]
    context = state["retrieved_docs"][-1] if state.get("retrieved_docs") else "No context"
    quality_score = state.get("retrieval_quality_score", 0)

    # Adjust system prompt based on retrieval quality
    if quality_score > 0.8:
        quality_instruction = "The retrieved documents are highly relevant. Answer based on them."
    elif quality_score > 0.6:
        quality_instruction = "The retrieved documents are somewhat relevant. Use them but note any gaps."
    else:
        quality_instruction = "The retrieved documents may not fully answer the question. Acknowledge limitations."

    system_prompt = f"""You are a helpful AI assistant. {quality_instruction}

Rules:
1. Answer only based on the provided context
2. Be honest about limitations
3. Cite sources when relevant
4. Confidence should match retrieval quality"""

    user_message = f"""Context:
{context}

Question: {question}

Answer:"""

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
    Evaluate answer considering retrieval quality AND context sufficiency.

    Phase 6 Enhancement: Added pre-generation context completeness checks
    to detect insufficient retrieval early.
    """
    question = state["original_query"]
    answer = state.get("final_answer", "")
    retrieval_quality = state.get("retrieval_quality_score", 0)
    context = "\n\n".join(state.get("retrieved_docs", [""]))

    # ============ PHASE 6: CONTEXT SUFFICIENCY CHECK ============
    # Check if retrieved context is sufficient BEFORE evaluating answer
    context_sufficiency_prompt = f"""Retrieved Context:
{context[:2000]}

Question:
{question}

Evaluate the completeness of the retrieved context:
1. Does context contain ALL information needed to answer the question?
2. Are there missing key details, facts, or aspects?
3. What is your confidence that the context is complete (0.0-1.0)?

Response as JSON:
{{
    "is_sufficient": true/false,
    "missing_aspects": ["aspect 1", "aspect 2"],
    "sufficiency_score": 0.0-1.0,
    "reasoning": "brief explanation of what is present or missing"
}}"""

    sufficiency_response = llm.invoke(context_sufficiency_prompt)

    # Parse context sufficiency evaluation
    try:
        json_match = re.search(r'\{.*\}', sufficiency_response.content, re.DOTALL)
        sufficiency_eval = json.loads(json_match.group()) if json_match else {}
    except:
        # Assume sufficient if parsing fails (fail open)
        sufficiency_eval = {
            "is_sufficient": True,
            "sufficiency_score": 0.7,
            "missing_aspects": []
        }

    context_is_sufficient = sufficiency_eval.get("is_sufficient", True)
    sufficiency_score = sufficiency_eval.get("sufficiency_score", 0.7)
    missing_aspects = sufficiency_eval.get("missing_aspects", [])
    sufficiency_reasoning = sufficiency_eval.get("reasoning", "")

    # Log context sufficiency analysis
    if not context_is_sufficient or sufficiency_score < 0.6:
        print(f"\n{'='*60}")
        print(f"CONTEXT INSUFFICIENCY DETECTED")
        print(f"Sufficiency Score: {sufficiency_score:.0%}")
        print(f"Is Sufficient: {context_is_sufficient}")
        print(f"Missing Aspects ({len(missing_aspects)}):")
        for aspect in missing_aspects:
            print(f"  - {aspect}")
        print(f"Reasoning: {sufficiency_reasoning}")
        print(f"Action: Will inform answer evaluation")
        print(f"{'='*60}\n")

    # ============ ORIGINAL ANSWER EVALUATION ============
    # Lower the sufficiency threshold if retrieval OR context was poor
    quality_threshold = 0.5 if (retrieval_quality < 0.6 or sufficiency_score < 0.6) else 0.65

    evaluation_prompt = f"""Question: {question}
Answer: {answer}
Retrieval quality: {retrieval_quality:.0%}
Context sufficiency: {sufficiency_score:.0%}

Evaluate this answer:
1. Relevant? (yes/no)
2. Complete? (yes/no)
3. Accurate? (yes/no)
4. Confidence (0-100)

Note: Context sufficiency of {sufficiency_score:.0%} should inform your confidence assessment.

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
        # Phase 6: Context sufficiency fields
        "context_sufficiency_score": sufficiency_score,
        "context_is_sufficient": context_is_sufficient,
        "missing_context_aspects": missing_aspects,

        # Original answer evaluation fields
        "is_answer_sufficient": is_sufficient,
        "confidence_score": confidence,
        "messages": [AIMessage(content=f"Evaluation: Confidence={confidence:.0%}, Context={sufficiency_score:.0%}")],
    }
