from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


class DocumentScore(TypedDict):
    """Individual document relevance score with explanation"""
    document_id: str           # Document identifier (e.g., "doc_0", "doc_1")
    relevance_score: float     # 0-100 relevance score
    reasoning: str             # Brief explanation for this document's score


class RankingResult(TypedDict):
    """Structured output schema for document reranking"""
    scored_documents: list[DocumentScore]  # Per-document scores with reasoning
    overall_reasoning: str  # Brief summary of overall ranking approach


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

        self.llm = ChatOpenAI(
            model=spec.name,
            temperature=spec.temperature,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity
        )
        self.structured_llm = self.llm.with_structured_output(RankingResult)

    def rank(self, query: str, documents: list[Document], fallback_scores: list[float] = None) -> list[tuple[Document, float]]:
        """
        Rank documents using LLM with metadata-aware scoring.

        Args:
            query: Search query
            documents: Documents to rank (typically pre-ranked by CrossEncoder)
            fallback_scores: Optional fallback scores (e.g., from CrossEncoder) to use if LLM fails

        Returns:
            List of (document, score) tuples sorted by relevance, limited to top_k
        """
        if not documents:
            return []

        from advanced_agentic_rag_langgraph.prompts import get_prompt

        # Create document ID mapping for robust parsing
        doc_id_map = {}
        doc_list = []

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc_id_map[doc_id] = i

            meta = doc.metadata
            doc_context = f"{doc_id}: "

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
            doc_context += f"\n   Content: {doc.page_content[:1000]}"

            doc_list.append(doc_context)

        ranking_prompt = get_prompt("llm_reranking", query=query, doc_list='\n'.join(doc_list))

        # Prepare fallback scores (preserve CrossEncoder ranking order if available)
        if fallback_scores is None:
            # If no fallback provided, use position-based scores (earlier = higher)
            # This preserves input ranking order (e.g., from CrossEncoder stage)
            fallback_scores = [100 - (i * 5) for i in range(len(documents))]

        try:
            result = self.structured_llm.invoke([HumanMessage(content=ranking_prompt)])
            scored_docs = result["scored_documents"]

            # Create score mapping using document_id strings for robustness
            score_map = {}
            for doc_score in scored_docs:
                doc_id = doc_score["document_id"]
                if doc_id in doc_id_map:
                    idx = doc_id_map[doc_id]
                    score_map[idx] = doc_score["relevance_score"]
                else:
                    print(f"Warning: Invalid document_id '{doc_id}' in LLM response (expected: {list(doc_id_map.keys())})")

            # If incomplete scoring, fall back to CrossEncoder ranking entirely
            if len(score_map) != len(documents):
                missing = [f"doc_{i}" for i in range(len(documents)) if i not in score_map]
                print(f"Warning: Incomplete LLM scoring ({len(score_map)}/{len(documents)} docs scored, missing: {missing})")
                print(f"Falling back to CrossEncoder ranking (preserves stage 1 results)")
                scores = fallback_scores
            else:
                # Complete scoring - use LLM scores
                scores = [score_map[i] for i in range(len(documents))]

        except Exception as e:
            print(f"Warning: Reranking failed: {e}. Using fallback scores (preserving CrossEncoder ranking).")
            scores = fallback_scores

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:self.top_k]
