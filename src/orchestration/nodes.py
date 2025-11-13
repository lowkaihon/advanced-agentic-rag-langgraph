from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.retrieval import (
    expand_query,
    rewrite_query,
    HybridRetriever,
    ReRanker,
    SemanticRetriever,
)
from src.core import setup_retriever
import re
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
hybrid_retriever = None  # Will be initialized via setup_retriever()

# ============ NEW: QUERY OPTIMIZATION STAGE ============

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
    """Decide which retrieval strategy to use based on query"""

    query = state["current_query"]

    strategy_prompt = f"""For this query: "{query}"

Which retrieval strategy would work best?
- "semantic": Complex questions needing understanding
- "keyword": Specific factual lookups
- "hybrid": Mixed or uncertain

Return JSON:
{{"strategy": "semantic" | "keyword" | "hybrid"}}"""

    response = llm.invoke(strategy_prompt)

    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {"strategy": "hybrid"}
        strategy = data.get("strategy", "hybrid")
    except:
        strategy = "hybrid"

    return {
        "retrieval_strategy": strategy,
        "messages": [AIMessage(content=f"Strategy chosen: {strategy}")],
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
        "messages": [ToolMessage(content=f"Retrieved {len(unique_docs)} documents")],
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
