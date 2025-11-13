"""
Strategy selection module for intelligent retrieval strategy selection.

Combines:
- Document profiling (corpus characteristics)
- Query analysis (query features)
- Heuristic rules (fast, deterministic)
- LLM reasoning (flexible, handles edge cases)
"""

import re
from typing import Dict, Literal, Tuple
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

    def _llm_tiebreaker(
        self,
        query: str,
        query_features: Dict,
        corpus_stats: Dict,
        scores: Dict[str, float]
    ) -> Tuple[Literal["semantic", "keyword", "hybrid"], str]:
        """
        Use LLM to break ties or handle ambiguous cases.

        Args:
            query: Original query
            query_features: Extracted query features
            corpus_stats: Corpus statistics
            scores: Current heuristic scores

        Returns:
            Tuple of (strategy, reasoning)
        """
        # Format corpus info
        corpus_info = ""
        if corpus_stats:
            corpus_info = f"""
Corpus characteristics:
- Total documents: {corpus_stats.get('total_documents', 'unknown')}
- Avg technical density: {corpus_stats.get('avg_technical_density', 'unknown')}
- Document types: {corpus_stats.get('document_types', {})}
"""

        # Format query features
        features_info = f"""
Query features:
- Intent: {query_features['intent']}
- Question type: {query_features['question_type']}
- Technical: {'Yes' if query_features['has_technical_terms'] else 'No'}
- Complexity: {query_features['complexity']}
"""

        # Format current scores
        scores_info = "\n".join([f"- {k}: {v:.2f}" for k, v in scores.items()])

        # Create prompt
        prompt = f"""Given this query: "{query}"

{features_info}
{corpus_info}

Current heuristic scores:
{scores_info}

Choose the best retrieval strategy:
- **semantic**: Best for conceptual questions, understanding meaning, finding similar concepts
- **keyword**: Best for exact lookups, technical terms, specific factual queries
- **hybrid**: Best for mixed queries, comparisons, or uncertain cases

Return ONLY the strategy name (semantic, keyword, or hybrid) followed by a brief reason.
Format: <strategy>: <reason>

Strategy:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip().lower()

        # Parse response
        for strategy in ["semantic", "keyword", "hybrid"]:
            if strategy in result:
                reason = result.replace(strategy, "").strip(":").strip()
                return strategy, f"LLM decision: {reason}"

        # Fallback to highest heuristic score
        strategy = max(scores, key=scores.get)
        return strategy, "LLM decision inconclusive, used heuristic"

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
