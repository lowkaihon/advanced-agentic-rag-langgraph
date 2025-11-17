from langchain_openai import ChatOpenAI
import json
import re

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def expand_query(query: str) -> list[str]:
    """Generate multiple query variations for better retrieval"""

    expansion_prompt = f"""For this question: "{query}"

Generate 3 alternative phrasings that would help retrieve better information:
- One that emphasizes technical implementation and mechanisms
- One that focuses on practical applications and use cases
- One that targets underlying concepts and principles

Guidelines:
- Preserve all technical terms, acronyms, and proper nouns EXACTLY as written
- Each variation should maintain the original meaning
- Variations should cover different aspects or perspectives

Return as JSON:
{{
    "variations": [
        "variation 1",
        "variation 2",
        "variation 3"
    ]
}}

Return ONLY the JSON, no explanation."""

    try:
        response = llm.invoke(expansion_prompt)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)

        if not json_match:
            print(f"Warning: No JSON found in expansion response")
            return [query]

        data = json.loads(json_match.group())
        variations = data.get("variations", [])

        # Validate variations
        valid_variations = [v for v in variations if v and isinstance(v, str) and v != query]

        return [query] + valid_variations

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Warning: Query expansion failed: {e}")
        return [query]


def rewrite_query(query: str, retrieval_context: str = None) -> str:
    """Rewrite unclear queries for better retrieval accuracy"""

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

    response = llm.invoke(rewrite_prompt)
    return response.content.strip().strip('"\'')  # Strip quotes if LLM added them


def optimize_query_for_strategy(
    query: str,
    new_strategy: str,
    old_strategy: str,
    issues: list[str] = None
) -> str:
    """
    Optimize query for a specific retrieval strategy (keyword/semantic/hybrid).

    Research-backed pattern from CRAG and PreQRAG:
    - Keyword (BM25): Emphasize specific terms, proper nouns, technical identifiers
    - Semantic (vector): Broaden to conceptual phrasing, semantic relationships
    - Hybrid: Balance specificity with conceptual framing

    Args:
        query: Current query to optimize
        new_strategy: Target strategy ("keyword", "semantic", or "hybrid")
        old_strategy: Previous strategy that didn't work well
        issues: List of quality issues detected (e.g., "off_topic", "missing_key_info")

    Returns:
        Optimized query string tailored to new strategy
    """

    # Build issue context if available
    issue_context = ""
    if issues:
        issue_context = f"\n\nIssues with previous retrieval:\n- " + "\n- ".join(issues)

    # Strategy-specific optimization guidance
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

    optimization_prompt = f"""You are optimizing a query for {new_strategy} retrieval after {old_strategy} retrieval did not work well.

Original query: "{query}"{issue_context}

{strategy_guidance.get(new_strategy, strategy_guidance["hybrid"])}

TASK: Rewrite the query to be optimized for {new_strategy} retrieval while preserving the user's intent.

CRITICAL GUIDELINES:
- Preserve all technical terms, acronyms, and proper nouns EXACTLY as written
- Keep the core user intent unchanged
- Adjust phrasing and terminology to match {new_strategy} characteristics
- Be concise but complete
- Return ONLY the optimized query, no explanation"""

    response = llm.invoke(optimization_prompt)
    optimized = response.content.strip().strip('"\'')

    print(f"\n{'='*60}")
    print(f"STRATEGY-SPECIFIC QUERY OPTIMIZATION")
    print(f"Strategy switch: {old_strategy} -> {new_strategy}")
    print(f"Original query: {query}")
    print(f"Optimized query: {optimized}")
    print(f"{'='*60}\n")

    return optimized
