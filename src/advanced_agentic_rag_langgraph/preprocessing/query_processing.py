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

        # Format conversation history (last 3 turns)
        history_text = self._format_history(conversation_history[-3:])

        # Create prompt for LLM
        prompt = f"""Given this conversation history:

{history_text}

The user just asked: "{query}"

Task: If the query references previous context (pronouns like "it", "that", "they", implicit references, or follow-up questions), rewrite it to be self-contained and clear by adding necessary context from the conversation history. Preserve all technical terms, acronyms, and proper nouns exactly as written. If the query is already self-contained, return it unchanged.

Return ONLY the rewritten or original query, nothing else."""

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