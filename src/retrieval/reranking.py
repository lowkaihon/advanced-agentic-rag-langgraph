from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import json
import re


class ReRanker:
    """Rerank documents using LLM-as-Judge pattern"""

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def rank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """Rerank documents using relevance scoring"""

        if not documents:
            return []

        # Build ranking prompt
        doc_list = "\n".join([
            f"{i+1}. [{doc.metadata.get('source', 'unknown')}] {doc.page_content[:200]}..."
            for i, doc in enumerate(documents)
        ])

        ranking_prompt = f"""Query: "{query}"

Documents:
{doc_list}

For each document, rate its relevance to the query (0-100).
- 0-30: Not relevant
- 31-60: Somewhat relevant
- 61-85: Relevant
- 86-100: Highly relevant

Return as JSON:
{{
    "scores": [score1, score2, score3, ...]
}}

Return ONLY the JSON."""

        response = self.llm.invoke(ranking_prompt)

        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {"scores": [50]*len(documents)}
            scores = data.get("scores", [50]*len(documents))
        except:
            scores = [50] * len(documents)

        # Pair documents with scores and sort
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:self.top_k]
