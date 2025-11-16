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
