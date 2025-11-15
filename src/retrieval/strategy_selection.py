"""
Strategy selection module for intelligent retrieval strategy selection.

Combines:
- Document profiling (corpus characteristics)
- Query analysis (query features)
- Heuristic rules (fast, deterministic)
- LLM reasoning (flexible, handles edge cases)
"""

import re
from typing import Dict, Literal, Tuple, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class QueryAnalyzer:
    """
    Analyzes queries to extract features for strategy selection.

    Features extracted:
    - Length (word count)
    - Question type (what/how/why/when/where/who)
    - Intent (factual/conceptual/comparative/procedural)
    - Technical indicators (code, jargon, specific terms)
    - Specificity (general vs specific)
    """

    # Question word patterns
    QUESTION_TYPES = {
        'what': r'\bwhat\b',
        'how': r'\bhow\b',
        'why': r'\bwhy\b',
        'when': r'\bwhen\b',
        'where': r'\bwhere\b',
        'who': r'\bwho\b',
        'which': r'\bwhich\b',
    }

    # Technical indicators
    TECHNICAL_PATTERNS = {
        'code_syntax': r'[(){}\[\]<>]|->|=>|\.\w+\(',
        'function_call': r'\w+\([^)]*\)',
        'quoted_term': r'"[^"]+"',
        'camelCase': r'\b[a-z]+[A-Z]\w+\b',
        'snake_case': r'\b\w+_\w+\b',
        'acronym': r'\b[A-Z]{2,}\b',
    }

    def analyze(self, query: str) -> Dict:
        """
        Analyze query and extract features.

        Args:
            query: User query to analyze

        Returns:
            Dictionary of query features
        """
        query_lower = query.lower()
        words = query.split()

        # Basic metrics
        word_count = len(words)
        char_count = len(query)

        # Question type
        question_type = self._detect_question_type(query_lower)

        # Intent classification
        intent = self._classify_intent(query_lower, question_type)

        # Technical indicators
        technical_score = self._calculate_technical_score(query)
        has_technical_terms = technical_score > 0.3

        # Specificity check
        has_quoted_terms = '"' in query or "'" in query
        has_exact_match_intent = has_quoted_terms or self._has_exact_match_keywords(query_lower)

        # Complexity
        complexity = self._estimate_complexity(word_count, question_type, intent)

        return {
            "word_count": word_count,
            "char_count": char_count,
            "question_type": question_type,
            "intent": intent,
            "technical_score": round(technical_score, 2),
            "has_technical_terms": has_technical_terms,
            "has_quoted_terms": has_quoted_terms,
            "has_exact_match_intent": has_exact_match_intent,
            "complexity": complexity,
        }

    def _detect_question_type(self, query_lower: str) -> str:
        """Detect question word (what/how/why, etc.)"""
        for q_type, pattern in self.QUESTION_TYPES.items():
            if re.search(pattern, query_lower):
                return q_type
        return "statement"  # Not a question

    def _classify_intent(self, query_lower: str, question_type: str) -> Literal["factual", "conceptual", "comparative", "procedural"]:
        """
        Classify user intent based on query patterns.

        - factual: Looking for specific facts/definitions
        - conceptual: Understanding concepts/explanations
        - comparative: Comparing options
        - procedural: How-to/step-by-step
        """
        # Procedural: how-to questions
        if question_type == "how" or any(word in query_lower for word in ["step", "tutorial", "guide"]):
            return "procedural"

        # Comparative: comparison keywords
        if any(word in query_lower for word in ["vs", "versus", "compare", "difference", "better", "best"]):
            return "comparative"

        # Conceptual: why/what is/explain
        if question_type in ["why", "what"] or any(word in query_lower for word in ["explain", "understand", "concept"]):
            return "conceptual"

        # Default: factual
        return "factual"

    def _calculate_technical_score(self, query: str) -> float:
        """Calculate how technical the query is (0.0 to 1.0)"""
        total_patterns = len(self.TECHNICAL_PATTERNS)
        matches = 0

        for name, pattern in self.TECHNICAL_PATTERNS.items():
            if re.search(pattern, query):
                matches += 1

        return matches / total_patterns if total_patterns > 0 else 0.0

    def _has_exact_match_keywords(self, query_lower: str) -> bool:
        """Check for keywords suggesting exact match search"""
        exact_match_keywords = [
            "error code", "function", "class", "method",
            "parameter", "api", "command", "syntax"
        ]
        return any(keyword in query_lower for keyword in exact_match_keywords)

    def _estimate_complexity(
        self,
        word_count: int,
        question_type: str,
        intent: str
    ) -> Literal["simple", "moderate", "complex"]:
        """Estimate query complexity"""
        # Short queries are simple
        if word_count < 5:
            return "simple"

        # Conceptual/comparative questions are complex
        if intent in ["conceptual", "comparative"]:
            return "complex"

        # Long queries are complex
        if word_count > 12:
            return "complex"

        # Everything else is moderate
        return "moderate"


class StrategyDecision(TypedDict):
    """Structured output schema for strategy selection"""
    strategy: Literal["semantic", "keyword", "hybrid"]
    confidence: float  # 0.0-1.0
    reasoning: str
    corpus_match_score: float  # How well does strategy fit corpus characteristics?


class StrategySelector:
    """
    Selects retrieval strategy using hybrid approach:
    1. Heuristic rules (fast, deterministic)
    2. LLM reasoning (flexible, handles edge cases)

    Strategy selection considers:
    - Query features (from QueryAnalyzer)
    - Corpus characteristics (from DocumentProfiler)
    - Historical performance (optional, future enhancement)
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize strategy selector.

        Args:
            model: OpenAI model for LLM fallback
            temperature: Temperature for LLM
        """
        self.analyzer = QueryAnalyzer()
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(StrategyDecision)

    def select_strategy(
        self,
        query: str,
        corpus_stats: Dict = None
    ) -> Tuple[Literal["semantic", "keyword", "hybrid"], float, str]:
        """
        Select optimal retrieval strategy.

        Args:
            query: User query
            corpus_stats: Corpus-level statistics (optional)

        Returns:
            Tuple of (strategy, confidence, reasoning)
        """
        # Analyze query
        query_features = self.analyzer.analyze(query)

        # Apply heuristic rules
        scores, rule_activations = self._apply_heuristics(query_features, corpus_stats or {})

        # Get top strategy and confidence
        strategy = max(scores, key=scores.get)
        confidence = scores[strategy]

        # If confidence is low, use LLM for tiebreaker
        if confidence < 0.7:
            strategy, reasoning = self._llm_tiebreaker(query, query_features, corpus_stats, scores)
            confidence = 0.75  # LLM decision has moderate confidence
        else:
            reasoning = f"Heuristic rules (confidence: {confidence:.0%}). " + ", ".join(rule_activations)

        return strategy, confidence, reasoning

    def _apply_heuristics(
        self,
        query_features: Dict,
        corpus_stats: Dict
    ) -> Tuple[Dict[str, float], list[str]]:
        """
        Apply heuristic rules to calculate strategy scores.

        Returns:
            Tuple of (scores_dict, rule_activation_list)
        """
        scores = {
            "semantic": 0.0,
            "keyword": 0.0,
            "hybrid": 0.0,
        }
        rule_activations = []

        # ===== RULE 1: Quoted terms favor keyword =====
        if query_features["has_quoted_terms"]:
            scores["keyword"] += 0.35
            rule_activations.append("quoted terms->keyword")

        # ===== RULE 2: Exact match intent favors keyword =====
        if query_features["has_exact_match_intent"]:
            scores["keyword"] += 0.25
            rule_activations.append("exact match->keyword")

        # ===== RULE 3: Technical queries in technical corpus favor keyword =====
        if (query_features["has_technical_terms"] and
            corpus_stats.get("avg_technical_density", 0) > 0.5):
            scores["keyword"] += 0.3
            scores["hybrid"] += 0.15
            rule_activations.append("technical query+corpus->keyword/hybrid")

        # ===== RULE 4: Conceptual questions favor semantic =====
        if query_features["intent"] == "conceptual":
            scores["semantic"] += 0.35
            rule_activations.append("conceptual->semantic")

        # ===== RULE 5: How/Why questions favor semantic =====
        if query_features["question_type"] in ["how", "why"]:
            scores["semantic"] += 0.3
            rule_activations.append("how/why->semantic")

        # ===== RULE 6: Procedural questions favor hybrid =====
        if query_features["intent"] == "procedural":
            scores["hybrid"] += 0.4
            rule_activations.append("procedural->hybrid")

        # ===== RULE 7: Comparative questions favor hybrid =====
        if query_features["intent"] == "comparative":
            scores["hybrid"] += 0.35
            rule_activations.append("comparative->hybrid")

        # ===== RULE 8: Short queries (<5 words) favor semantic =====
        if query_features["word_count"] < 5:
            scores["semantic"] += 0.2
            rule_activations.append("short query->semantic")

        # ===== RULE 9: Complex queries favor hybrid =====
        if query_features["complexity"] == "complex":
            scores["hybrid"] += 0.25
            rule_activations.append("complex->hybrid")

        # ===== RULE 10: Factual lookups favor keyword =====
        if query_features["intent"] == "factual" and query_features["word_count"] < 8:
            scores["keyword"] += 0.25
            rule_activations.append("factual lookup->keyword")

        # Normalize scores to confidences (0-1 range)
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        else:
            # No rules activated - default to hybrid with low confidence
            scores["hybrid"] = 0.5

        return scores, rule_activations

    def _build_corpus_context(self, corpus_stats: Dict) -> str:
        """Build corpus-aware context for system prompt"""
        if not corpus_stats:
            return "No corpus statistics available."

        doc_types = corpus_stats.get('document_types', {})
        domains = corpus_stats.get('domain_distribution', {})
        avg_tech_density = corpus_stats.get('avg_technical_density', 0)

        corpus_type = "technical" if avg_tech_density > 0.6 else "general audience"
        primary_domain = list(domains.keys())[0] if domains else "general"

        context = f"""CORPUS PROFILE:
- Total documents: {corpus_stats.get('total_documents', 'unknown')}
- Type: {corpus_type} (technical density: {avg_tech_density:.0%})
- Document types: {', '.join(f"{k}={v}" for k, v in doc_types.items())}
- Primary domains: {', '.join(list(domains.keys())[:3])}
- Contains code: {corpus_stats.get('pct_with_code', 0):.0f}%
- Contains math: {corpus_stats.get('pct_with_math', 0):.0f}%

This corpus is {corpus_type} focused on {primary_domain} topics."""

        return context

    def _llm_tiebreaker(
        self,
        query: str,
        query_features: Dict,
        corpus_stats: Dict,
        scores: Dict[str, float]
    ) -> Tuple[Literal["semantic", "keyword", "hybrid"], str]:
        """
        Use LLM with structured output to break ties or handle ambiguous cases.

        Args:
            query: Original query
            query_features: Extracted query features
            corpus_stats: Corpus statistics
            scores: Current heuristic scores

        Returns:
            Tuple of (strategy, reasoning)
        """
        # Build corpus-aware system prompt
        corpus_context = self._build_corpus_context(corpus_stats or {})

        system_prompt = f"""You are a retrieval strategy selector optimized for this specific corpus.

{corpus_context}

STRATEGY SELECTION GUIDELINES:
1. **Semantic Search**: Best for understanding concepts, finding similar ideas, conceptual queries
   - Good when: corpus has research papers/conceptual content, query asks "why/how/what is"
   - Example: "What is attention mechanism?" uses semantic (find explanations)

2. **Keyword Search**: Best for exact lookups, technical terms, API names, specific citations
   - Good when: corpus has API references/code, query has quoted terms or specific names
   - Example: "transformer architecture attention function" uses keyword (exact terms)

3. **Hybrid Search**: Best for mixed content, comparisons, uncertain cases
   - Good when: corpus is diverse, query needs both concepts and exact matches
   - Example: "How does attention compare to RNN?" uses hybrid (concepts + exact names)

Consider corpus composition when selecting strategy:
- High math/research corpus: semantic works well for conceptual queries
- High code/API corpus: keyword works well for technical lookups
- Mixed corpus: hybrid provides best coverage

Provide:
- strategy: Your choice (semantic, keyword, or hybrid)
- confidence: 0.0-1.0 (how certain are you?)
- reasoning: Why this strategy for this corpus + query combination?
- corpus_match_score: 0.0-1.0 (how well does strategy fit corpus characteristics?)
"""

        # Build query context
        query_context = f"""Query: "{query}"

Query characteristics:
- Intent: {query_features['intent']}
- Question type: {query_features['question_type']}
- Complexity: {query_features['complexity']}
- Has technical terms: {query_features['has_technical_terms']}
- Has quoted terms: {query_features['has_quoted_terms']}
- Has exact match intent: {query_features['has_exact_match_intent']}

Current heuristic scores (for reference):
- Semantic: {scores.get('semantic', 0):.2f}
- Keyword: {scores.get('keyword', 0):.2f}
- Hybrid: {scores.get('hybrid', 0):.2f}

Select the optimal strategy considering both query characteristics and corpus composition.
"""

        try:
            # Use structured output for reliable parsing
            decision = self.structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_context}
            ])

            return decision['strategy'], f"LLM decision (confidence: {decision['confidence']:.0%}, corpus match: {decision['corpus_match_score']:.0%}): {decision['reasoning']}"

        except Exception as e:
            # Fallback to highest heuristic score if LLM fails
            print(f"Warning: LLM tiebreaker failed: {e}")
            strategy = max(scores, key=scores.get)
            return strategy, f"Heuristic fallback (LLM failed): {strategy} scored highest ({scores[strategy]:.2f})"

    def explain_decision(
        self,
        query: str,
        corpus_stats: Dict = None
    ) -> str:
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
        query_features = self.analyzer.analyze(query)

        explanation = f"""
Strategy Selection for: "{query}"
{'='*60}

Query Analysis:
- Word count: {query_features['word_count']}
- Question type: {query_features['question_type']}
- Intent: {query_features['intent']}
- Complexity: {query_features['complexity']}
- Technical: {'Yes' if query_features['has_technical_terms'] else 'No'}
- Exact match intent: {'Yes' if query_features['has_exact_match_intent'] else 'No'}

Selected Strategy: {strategy.upper()}
Confidence: {confidence:.0%}

Reasoning:
{reasoning}
{'='*60}
"""
        return explanation.strip()
