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
    """Generate query variations for multi-query retrieval."""
    expansion_prompt = f"""Generate 3 alternative phrasings for a question to help retrieve better information.

DETECT QUERY TYPE FIRST:

For COMPARISON queries ("X vs Y", "difference between X and Y", "how does X differ from Y"):
- Variation 1: Focus on understanding X independently
- Variation 2: Focus on understanding Y independently
- Variation 3: Focus on the relationship/comparison between X and Y

For ADAPTATION queries ("how does X adapt Y", "how does X modify Y for Z"):
- Variation 1: Focus on the original system (Y) being adapted
- Variation 2: Focus on the adapted system (X) and its changes
- Variation 3: Focus on the adaptation mechanism itself

For OTHER queries (factual, conceptual, how-to):
- Variation 1: Technical implementation and mechanisms
- Variation 2: Practical applications and use cases
- Variation 3: Underlying concepts and principles

EXAMPLE - COMPARISON:
Question: "What is the difference between SQL and NoSQL databases?"
Variations:
1. What are the core characteristics and data model of SQL relational databases?
2. What are the core characteristics and data model of NoSQL databases?
3. How do SQL and NoSQL databases differ in structure, scalability, and use cases?

EXAMPLE - ADAPTATION:
Question: "How does mobile-first design adapt responsive web design?"
Variations:
1. What is responsive web design and its core principles?
2. What is mobile-first design and how does it prioritize mobile devices?
3. What changes does mobile-first design make to the responsive design approach?

EXAMPLE - OTHER:
Question: "How does caching improve performance?"
Variations:
1. What are the technical mechanisms and implementation details of caching systems?
2. What are practical use cases where caching provides performance benefits?
3. What are the underlying principles of how caching reduces latency and load?

NOW GENERATE FOR:
Question: "{query}"

Guidelines:
- Preserve all technical terms, acronyms, and proper nouns EXACTLY as written
- For comparison/adaptation queries, ensure variations retrieve BOTH sides independently
- Each variation should be independently useful for retrieval"""

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


def rewrite_query(query: str, keywords: list[str]) -> str:
    """
    Inject keywords into query for improved retrieval.

    Args:
        query: Original query to enhance
        keywords: List of specific, non-tautological keywords to incorporate

    Returns:
        Rewritten query with keywords naturally incorporated
    """
    if not keywords:
        return query

    rewrite_prompt = f"""Original query: {query}

Keywords to incorporate: {', '.join(keywords)}

Rewrite the query to naturally include these keywords while preserving the original intent.
Keep it concise (similar length to original). Output only the rewritten query."""

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
- "platform features" -> "specific platform capabilities and feature implementations"
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
- "system architecture" -> "understanding how the system is structured and deployed"
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
- "authentication system" -> "authentication mechanism: how it works and implementation details"
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
