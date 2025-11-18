from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


class RankingResult(TypedDict):
    """Structured output schema for document reranking"""
    scores: list[float]  # 0-100 for each document
    reasoning: str  # Brief explanation of ranking decisions


class LLMMetadataReRanker:
    """Rerank documents using LLM-as-Judge with metadata-aware relevance scoring."""

    def __init__(self, top_k: int = 4):
        """
        Initialize LLM metadata reranker with tier-based model configuration.

        Args:
            top_k: Number of top documents to return after reranking
        """
        self.top_k = top_k

        spec = get_model_for_task("llm_reranking")
        model_kwargs = {}
        if spec.reasoning_effort:
            model_kwargs["reasoning_effort"] = spec.reasoning_effort

        self.llm = ChatOpenAI(
            model=spec.name,
            temperature=spec.temperature,
            model_kwargs=model_kwargs
        )
        self.structured_llm = self.llm.with_structured_output(RankingResult)

    def rank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        if not documents:
            return []

        from advanced_agentic_rag_langgraph.prompts import get_prompt

        doc_list = []
        for i, doc in enumerate(documents):
            meta = doc.metadata
            doc_context = f"{i+1}. "

            content_type = meta.get('content_type', 'unknown')
            tech_level = meta.get('technical_level', 'unknown')
            domain = meta.get('domain', 'general')
            doc_context += f"[Type: {content_type} | Level: {tech_level} | Domain: {domain}"

            if meta.get('has_math'):
                doc_context += " | Has math"
            if meta.get('has_code'):
                doc_context += " | Has code"

            source = meta.get('source', 'unknown')
            doc_context += f" | Source: {source}]"
            doc_context += f"\n   Content: {doc.page_content[:200]}..."

            doc_list.append(doc_context)

        ranking_prompt = get_prompt("llm_reranking", query=query, doc_list='\n'.join(doc_list))

        try:
            result = self.structured_llm.invoke([HumanMessage(content=ranking_prompt)])
            scores = result["scores"]

            if len(scores) != len(documents):
                print(f"Warning: LLM returned {len(scores)} scores for {len(documents)} documents. Using fallback.")
                scores = [50] * len(documents)

        except Exception as e:
            print(f"Warning: Reranking failed: {e}. Using neutral scores.")
            scores = [50] * len(documents)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:self.top_k]
