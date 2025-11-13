from langchain_openai import ChatOpenAI
import json
import re

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def expand_query(query: str) -> list[str]:
    """Generate multiple query variations for better retrieval"""

    expansion_prompt = f"""For this question: "{query}"

Generate 3 alternative phrasings that would help retrieve better information:
- One more technical/formal version
- One simpler/layman's version
- One focused on different aspect of the topic

Return as JSON:
{{
    "variations": [
        "variation 1",
        "variation 2",
        "variation 3"
    ]
}}

Return ONLY the JSON, no explanation."""

    response = llm.invoke(expansion_prompt)

    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {"variations": [query]}
        return [query] + data.get("variations", [])
    except:
        return [query]


def rewrite_query(query: str, retrieval_context: str = None) -> str:
    """Rewrite unclear queries for better retrieval accuracy"""

    context_info = ""
    if retrieval_context:
        context_info = f"\n\nPrevious retrieval returned insufficient results."

    rewrite_prompt = f"""Rewrite this query to be clearer and more specific for a search system:

Original: "{query}"{context_info}

Focus on:
- Adding missing context
- Using standard terminology
- Being concise but complete
- Removing ambiguity

Return ONLY the rewritten query."""

    response = llm.invoke(rewrite_prompt)
    return response.content.strip()
