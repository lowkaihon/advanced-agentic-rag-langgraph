"""
Query processing module for conversational query rewriting.

This module handles query rewriting using conversation history to make
queries self-contained and clear.
"""

from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class ConversationalRewriter:
    """
    Rewrites queries using conversation history to make them self-contained.

    Uses a simple LLM-based approach:
    - If query references previous context, rewrite it
    - If query is already clear, return unchanged
    - No complex rule-based logic - let LLM handle everything
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the conversational rewriter.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
            temperature: Temperature for LLM (0 for deterministic)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def rewrite(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, str]:
        """
        Rewrite query with conversation context if needed.

        Args:
            query: Current user query
            conversation_history: List of conversation turns
                                Each turn is dict with 'user' and 'assistant' keys

        Returns:
            Tuple of (rewritten_query, reasoning)
        """
        # Skip if no conversation history
        if not conversation_history or len(conversation_history) == 0:
            return query, "No conversation history - no rewrite needed"

        # Skip if query is already long/detailed (likely self-contained)
        if len(query.split()) > 12:
            return query, "Query is already detailed - no rewrite needed"

        # Format conversation history (last 3 turns)
        history_text = self._format_history(conversation_history[-3:])

        # Create prompt for LLM
        prompt = f"""Given this conversation history:

{history_text}

The user just asked: "{query}"

Task: If the query references previous context (pronouns like "it", "that", "they", implicit references, or follow-up questions), rewrite it to be self-contained and clear. Otherwise, return it exactly as-is.

IMPORTANT: Return ONLY the query (rewritten or original), nothing else. No explanations, no quotes.

Query:"""

        # Get LLM response
        response = self.llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()

        # Remove quotes if LLM added them
        rewritten = rewritten.strip('"\'')

        # Determine if rewrite happened
        if rewritten.lower() == query.lower():
            reasoning = "Query is already self-contained"
        else:
            reasoning = f"Rewritten to add context from conversation"

        return rewritten, reasoning

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for prompt.

        Args:
            history: List of conversation turns

        Returns:
            Formatted history string
        """
        if not history:
            return "(No previous conversation)"

        lines = []
        for i, turn in enumerate(history, 1):
            user_query = turn.get('user', '')
            assistant_response = turn.get('assistant', '')

            lines.append(f"Turn {i}:")
            lines.append(f"  User: {user_query}")

            # Truncate long responses
            if len(assistant_response) > 200:
                assistant_response = assistant_response[:200] + "..."

            lines.append(f"  Assistant: {assistant_response}")
            lines.append("")  # Empty line for readability

        return "\n".join(lines)

    def should_rewrite(self, query: str, conversation_history: List[Dict]) -> bool:
        """
        Heuristic check if rewriting might be needed.

        This is optional and can be used to skip the LLM call entirely
        for obviously self-contained queries.

        Args:
            query: User query
            conversation_history: Conversation history

        Returns:
            True if rewriting might be beneficial
        """
        # No history = no rewrite needed
        if not conversation_history:
            return False

        # Very short queries likely need context
        if len(query.split()) < 5:
            return True

        # Check for pronouns (common indicators of context dependency)
        pronouns = ['it', 'that', 'this', 'they', 'them', 'those', 'these']
        query_lower = query.lower()
        if any(pronoun in query_lower.split() for pronoun in pronouns):
            return True

        # Check for follow-up phrases
        followup_phrases = [
            'how about', 'what about', 'also', 'too',
            'as well', 'another', 'more', 'else'
        ]
        if any(phrase in query_lower for phrase in followup_phrases):
            return True

        # Check for incomplete questions (might indicate continuation)
        if query.endswith('?') and len(query.split()) < 7:
            return True

        # Otherwise, query is likely self-contained
        return False


class QueryExpander:
    """
    Expands queries into multiple variations for better retrieval coverage.

    Note: This duplicates functionality from src/retrieval/query_optimization.py
    but is included here for completeness of the preprocessing module.
    Consider using the existing expand_query function instead.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize query expander with LLM."""
        self.llm = ChatOpenAI(model=model, temperature=0.7)

    def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Expand query into multiple variations.

        Args:
            query: Original query
            num_variations: Number of variations to generate

        Returns:
            List of query variations including original
        """
        prompt = f"""Generate {num_variations - 1} alternative phrasings of this query for better search coverage:

Original query: "{query}"

Generate variations that:
1. Use different terminology but same meaning
2. Are more specific or more general
3. Focus on different aspects of the query

Return as a simple list, one per line, without numbers or explanations:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        variations = [line.strip() for line in response.content.strip().split('\n') if line.strip()]

        # Include original query
        return [query] + variations[:num_variations - 1]
