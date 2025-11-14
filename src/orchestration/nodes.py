from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.retrieval import (
    expand_query,
    rewrite_query,
    HybridRetriever,
    ReRanker,
    SemanticRetriever,
)
from src.retrieval.strategy_selection import StrategySelector
from src.core import setup_retriever, get_corpus_stats
from src.preprocessing.query_processing import ConversationalRewriter
import re
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
hybrid_retriever = None  # Will be initialized via setup_retriever()
conversational_rewriter = ConversationalRewriter()  # For query rewriting
strategy_selector = StrategySelector()  # For intelligent strategy selection

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

def query_expansion_node(state: dict) -> dict:
    """Generate query variations for comprehensive retrieval"""

    query = state["original_query"]
    expansions = expand_query(query)

    print(f"Original: {query}")
    print(f"Expansions: {expansions[1:]}")  # Skip the original

    return {
        "query_expansions": expansions,
        "current_query": query,  # Start with original
    }

def decide_retrieval_strategy_node(state: dict) -> dict:
    """
    Decide which retrieval strategy to use based on query and corpus characteristics.

    Uses hybrid heuristics + LLM fallback for intelligent strategy selection.
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
        from src.core import setup_retriever
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

    return {
        "retrieved_docs": [docs_text],
        "retrieval_quality_score": quality_score,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "unique_docs_list": unique_docs,  # Store Document objects for metadata analysis
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
        print(f"Rewritten: {query} → {rewritten}")

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

def evaluate_answer_with_retrieval_node(state: dict) -> dict:
    """Evaluate answer, considering retrieval quality"""

    question = state["original_query"]
    answer = state.get("final_answer", "")
    retrieval_quality = state.get("retrieval_quality_score", 0)

    # Lower the sufficiency threshold if retrieval was poor
    quality_threshold = 0.5 if retrieval_quality < 0.6 else 0.65

    evaluation_prompt = f"""Question: {question}
Answer: {answer}
Retrieval quality: {retrieval_quality:.0%}

Evaluate this answer:
1. Relevant? (yes/no)
2. Complete? (yes/no)
3. Accurate? (yes/no)
4. Confidence (0-100)

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
        "is_answer_sufficient": is_sufficient,
        "confidence_score": confidence,
        "messages": [AIMessage(content=f"Evaluation: Confidence={confidence:.0%}")],
    }
