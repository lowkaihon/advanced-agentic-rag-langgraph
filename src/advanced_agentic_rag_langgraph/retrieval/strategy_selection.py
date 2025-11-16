"""
Strategy selection using pure LLM classification.

Aligns with best practices: 'Dynamic Strategy Selection using lightweight classifiers'
Domain-agnostic design - works for any document type (technical docs, legal, medical, etc.)

Previous implementation used 10 heuristic rules with 9 regex-based metrics.
Simplified to pure LLM for better accuracy and maintainability.
"""

from typing import Dict, Literal, Tuple, TypedDict
from langchain_openai import ChatOpenAI


class StrategyDecision(TypedDict):
    """Structured output schema for strategy selection"""
    strategy: Literal["semantic", "keyword", "hybrid"]
    confidence: float
    reasoning: str


class StrategySelector:
    """
    Pure LLM-based retrieval strategy selection.

    Domain-agnostic - adapts to any corpus type through corpus_stats.
    Zero heuristics - LLM handles all edge cases and classification.

    Replaces previous 10-rule heuristic system with simpler, more accurate LLM approach.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize strategy selector.

        Args:
            model: OpenAI model for LLM classification
            temperature: Temperature for LLM (0 for deterministic)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(StrategyDecision)

    def select_strategy(
        self,
        query: str,
        corpus_stats: Dict = None
    ) -> Tuple[Literal["semantic", "keyword", "hybrid"], float, str]:
        """
        Select optimal retrieval strategy using LLM classification.

        Args:
            query: User query
            corpus_stats: Corpus characteristics (document types, technical density, etc.)

        Returns:
            Tuple of (strategy, confidence, reasoning)
        """
        corpus_context = self._build_corpus_context(corpus_stats or {})

        system_prompt = f"""You are a retrieval strategy selector for a RAG system.

{corpus_context}

STRATEGY SELECTION GUIDELINES:

**Semantic Search** (dense vector embeddings):
- Best for: Conceptual understanding, explanations, "why/how/what" questions
- Strengths: Finds semantically similar content, handles paraphrasing and synonyms
- Use when: User seeks understanding, concepts, or thematic information
- Examples:
  * "What is X?" (definitional)
  * "How does Y work?" (explanatory)
  * "Why use Z?" (conceptual reasoning)

**Keyword Search** (BM25 term matching):
- Best for: Exact term lookups, specific names, codes, identifiers, citations
- Strengths: Precise matching on exact terms, proper nouns, technical identifiers
- Use when: User needs specific information by exact name/term
- Examples:
  * API/function names lookup
  * Error codes or specific identifiers
  * Proper nouns (people, places, products)
  * Technical terminology with exact spelling

**Hybrid Search** (semantic + keyword combined):
- Best for: Comparisons, multi-faceted queries, queries needing both concepts AND exact terms
- Strengths: Combines understanding with precision, covers multiple retrieval modes
- Use when: Query has multiple aspects or unclear which mode is better
- Examples:
  * Comparisons ("Compare A vs B")
  * Procedural + specific ("How to use X feature?")
  * Queries with quoted terms AND conceptual elements
  * Ambiguous queries where both modes add value

IMPORTANT DECISION FACTORS:

1. **Query Intent**: What is the user trying to accomplish?
   - Understanding/learning -> semantic
   - Finding specific item -> keyword
   - Comparing/analyzing -> hybrid

2. **Match Requirements**:
   - Semantic similarity needed -> semantic or hybrid
   - Exact term matching needed -> keyword or hybrid
   - Both needed -> hybrid

3. **Corpus Characteristics**: Consider document type and content style
   - High technical density + technical query -> keyword or hybrid
   - Conceptual content + conceptual query -> semantic
   - Mixed corpus -> hybrid (safest)

4. **Quoted Terms**: Consider overall intent, not just presence of quotes
   - Quoted terms + lookup only -> keyword
   - Quoted terms + comparison -> hybrid
   - Quoted terms + conceptual question -> semantic or hybrid

Provide structured output with your reasoning."""

        query_prompt = f"""Query: "{query}"

Analyze this query and select the optimal retrieval strategy considering:
1. User intent (what are they trying to find?)
2. Whether exact matching or semantic similarity is more important
3. Corpus characteristics (if provided)"""

        try:
            decision = self.structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_prompt}
            ])

            return decision['strategy'], decision['confidence'], decision['reasoning']

        except Exception as e:
            print(f"Warning: LLM strategy selection failed: {e}")
            return "hybrid", 0.5, f"LLM failed, defaulting to hybrid: {e}"

    def _build_corpus_context(self, corpus_stats: Dict) -> str:
        """Build domain-agnostic corpus context for system prompt"""
        if not corpus_stats:
            return "CORPUS PROFILE: No statistics available (assume general-purpose corpus)"

        doc_types = corpus_stats.get('document_types', {})
        domains = corpus_stats.get('domain_distribution', {})
        avg_tech_density = corpus_stats.get('avg_technical_density', 0)

        # Domain-agnostic classification
        corpus_type = "technical" if avg_tech_density > 0.6 else "general"
        primary_domain = list(domains.keys())[0] if domains else "general"

        return f"""CORPUS PROFILE:
- Total documents: {corpus_stats.get('total_documents', 'unknown')}
- Content type: {corpus_type} (technical density: {avg_tech_density:.0%})
- Document types: {', '.join(f"{k}={v}" for k, v in doc_types.items())}
- Primary domains: {', '.join(list(domains.keys())[:3])}
- Contains code/technical content: {corpus_stats.get('pct_with_code', 0):.0f}%
- Contains mathematical notation: {corpus_stats.get('pct_with_math', 0):.0f}%

This is a {corpus_type} corpus focused on {primary_domain} topics."""

    def explain_decision(self, query: str, corpus_stats: Dict = None) -> str:
        """
        Get detailed explanation of strategy decision.

        Useful for debugging and demonstration.

        Args:
            query: User query
            corpus_stats: Corpus statistics

        Returns:
            Formatted explanation string
        """
        strategy, confidence, reasoning = self.select_strategy(query, corpus_stats)

        return f"""
Strategy Selection for: "{query}"
{'='*60}

Selected Strategy: {strategy.upper()}
Confidence: {confidence:.0%}

Reasoning:
{reasoning}
{'='*60}
""".strip()
