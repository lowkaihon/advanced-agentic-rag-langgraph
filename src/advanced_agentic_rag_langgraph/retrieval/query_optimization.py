from langchain_openai import ChatOpenAI
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from pydantic import BaseModel, Field


class QueryVariations(BaseModel):
    """Query expansion variations for multi-query retrieval."""
    variations: list[str] = Field(
        min_length=3,
        max_length=3,
        description="Three alternative phrasings of the query"
    )


def _get_expansion_llm():
    """Get LLM for query expansion with tier-based configuration."""
    spec = get_model_for_task("query_expansion")
    return ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )


def _get_rewriting_llm():
    """Get LLM for query rewriting with tier-based configuration."""
    spec = get_model_for_task("query_rewriting")
    return ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )


def _get_strategy_optimization_llm():
    """Get LLM for strategy-specific query optimization."""
    spec = get_model_for_task("strategy_optimization")
    return ChatOpenAI(
        model=spec.name,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        verbosity=spec.verbosity
    )


def expand_query(query: str) -> list[str]:
    """
    Generate query variations using structured output.

    Uses Pydantic schema for 95%+ parsing reliability vs 85-90% with regex.
    """
    expansion_prompt = f"""For this question: "{query}"

Generate 3 alternative phrasings that would help retrieve better information:
- One that emphasizes technical implementation and mechanisms
- One that focuses on practical applications and use cases
- One that targets underlying concepts and principles

Guidelines:
- Preserve all technical terms, acronyms, and proper nouns EXACTLY as written
- Each variation should maintain the original meaning
- Variations should cover different aspects or perspectives"""

    try:
        llm = _get_expansion_llm()
        structured_llm = llm.with_structured_output(QueryVariations)
        result = structured_llm.invoke(expansion_prompt)

        # Extract variations from Pydantic model
        variations = result.variations if hasattr(result, 'variations') else result.get('variations', [])
        valid_variations = [v for v in variations if v and isinstance(v, str) and v != query]

        return [query] + valid_variations

    except Exception as e:
        print(f"Warning: Query expansion failed: {e}")
        return [query]


def rewrite_query(query: str, retrieval_context: str = None) -> str:
    context_info = ""
    if retrieval_context:
        context_info = f"\n\nContext: {retrieval_context}"

    rewrite_prompt = f"""Rewrite this query to be clearer and more specific for a search system:

Original: "{query}"{context_info}

Guidelines:
- Preserve all technical terms, acronyms, and proper nouns EXACTLY as written
- Add missing context if the query is vague
- Use precise, unambiguous language
- Be concise but complete
- Remove unnecessary qualifiers

Return ONLY the rewritten query."""

    llm = _get_rewriting_llm()
    response = llm.invoke(rewrite_prompt)
    return response.content.strip().strip('"\'')



def optimize_query_for_strategy(
    query: str,
    strategy: str,
    old_strategy: str = None,
    issues: list[str] = None
) -> str:
    """
    Optimize query for specific retrieval strategy.

    Args:
        query: Query to optimize
        strategy: Target retrieval strategy (semantic/keyword/hybrid)
        old_strategy: Previous strategy (only when switching strategies)
        issues: Issues from previous retrieval (only when switching)

    Research-backed pattern from CRAG and PreQRAG.
    """
    issue_context = ""
    if issues:
        issue_context = f"\n\nIssues with previous retrieval:\n- " + "\n- ".join(issues)

    strategy_guidance = {
        "keyword": """
KEYWORD SEARCH (BM25) OPTIMIZATION:
- Add specific technical terms, identifiers, and proper nouns
- Include exact phrases that might appear in documents
- Use concrete terminology over abstract concepts
- Add acronyms, version numbers, or specific names
- Remove very broad or conceptual language
- Focus on term matching and specificity

Example transformations:
- "benefits" -> "specific quantitative benefits and advantages"
- "how it works" -> "implementation mechanism and technical process"
- "AI techniques" -> "machine learning algorithms and neural network architectures"
""",
        "semantic": """
SEMANTIC SEARCH (Vector) OPTIMIZATION:
- Broaden to conceptual and semantic relationships
- Use natural, contextual language
- Include related concepts and principles
- Add descriptive qualifiers and context
- Remove overly specific or narrow terms
- Focus on meaning and intent over exact terms

Example transformations:
- "BERT implementation" -> "understanding BERT architecture and how it's applied"
- "specific metrics" -> "approaches to measuring and evaluating performance"
- "version 2.0 features" -> "new capabilities and improvements in recent versions"
""",
        "hybrid": """
HYBRID SEARCH (BM25 + Vector) OPTIMIZATION:
- Balance specific terms with conceptual framing
- Include both exact terminology and descriptive context
- Combine technical identifiers with explanatory language
- Keep proper nouns but add conceptual qualifiers
- Aim for queries that work well with both matching styles

Example transformations:
- "transformer attention" -> "transformer attention mechanism: how it works and implementation details"
- "performance issues" -> "specific performance bottlenecks and optimization strategies"
"""
    }

    # Adjust prompt based on whether we're switching strategies
    if old_strategy:
        # Strategy switch scenario (early switch)
        prompt_context = f"You are optimizing a query for {strategy} retrieval after {old_strategy} retrieval did not work well."
        log_header = f"Strategy switch: {old_strategy} -> {strategy}"
    else:
        # Initial optimization or retry with same strategy
        prompt_context = f"You are optimizing a query for {strategy} retrieval."
        log_header = f"Strategy: {strategy}"

    optimization_prompt = f"""{prompt_context}

Original query: "{query}"{issue_context}

{strategy_guidance.get(strategy, strategy_guidance["hybrid"])}

TASK: Rewrite the query to be optimized for {strategy} retrieval while preserving the user's intent.

CRITICAL GUIDELINES:
- Preserve all technical terms, acronyms, and proper nouns EXACTLY as written
- Keep the core user intent unchanged
- Adjust phrasing and terminology to match {strategy} characteristics
- Be concise but complete
- Return ONLY the optimized query, no explanation"""

    llm = _get_strategy_optimization_llm()
    response = llm.invoke(optimization_prompt)
    optimized = response.content.strip().strip('"\'')

    print(f"\n{'='*60}")
    print(f"STRATEGY-SPECIFIC QUERY OPTIMIZATION")
    print(f"{log_header}")
    print(f"Original query: {query}")
    print(f"Optimized query: {optimized}")
    if issues:
        print(f"Issues triggering optimization: {', '.join(issues)}")
    print(f"{'='*60}\n")

    return optimized
