"""
LLM-based document reranking for Multi-Agent RAG merge stage.

Scores documents from multi-worker retrieval by relevance to the original question.
Uses same pattern as LLMMetadataReRanker: score each doc 0-100, sort, take top-k.

Key differences from single-hop LLMMetadataReRanker:
- Candidates come from multiple parallel workers
- No metadata scoring (workers already did two-stage reranking)
"""

from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


class DocumentScore(TypedDict):
    """Individual document relevance score with explanation."""
    document_id: str           # Document identifier (e.g., "doc_0", "doc_1")
    relevance_score: float     # 0-100 relevance score
    reasoning: str             # Brief explanation for this document's score


class RankingResult(TypedDict):
    """Structured output for document reranking."""
    scored_documents: list[DocumentScore]  # Per-document scores with reasoning
    overall_reasoning: str  # Brief summary of ranking approach


class MultiAgentMergeReRanker:
    """Score and rank documents from multi-worker retrieval results."""

    def __init__(self, top_k: int = 6):
        """
        Initialize relevance-based merge reranker.

        Args:
            top_k: Number of top documents to return after reranking
        """
        self.top_k = top_k

        spec = get_model_for_task("multi_agent_merge_reranking")

        self.llm = ChatOpenAI(
            model=spec.name,
            temperature=spec.temperature,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity,
        )
        self.structured_llm = self.llm.with_structured_output(RankingResult)

    def rerank(
        self,
        original_question: str,
        candidate_docs: list[Document],
        fallback_scores: list[float] = None,
    ) -> tuple[list[Document], list[float] | None]:
        """
        Score and rank documents by relevance to the original question.

        Args:
            original_question: The user's original question
            candidate_docs: Candidates from all workers
            fallback_scores: Fallback scores if LLM fails (position-based)

        Returns:
            Tuple of (top-k Documents, their scores 0-100).
            Scores is None if fallback was used (LLM failed or incomplete).
        """
        if not candidate_docs:
            return [], None

        if len(candidate_docs) <= self.top_k:
            return candidate_docs, None  # No reranking needed, no scores

        from advanced_agentic_rag_langgraph.prompts import get_prompt

        # Build document ID mapping (index-based for score lookup)
        doc_id_map = {f"doc_{i}": i for i, doc in enumerate(candidate_docs)}

        # Format documents for prompt
        doc_list = []
        for i, doc in enumerate(candidate_docs):
            doc_id = f"doc_{i}"
            source = doc.metadata.get("source", "unknown")

            content_preview = doc.page_content[:1000]
            doc_list.append(f"{doc_id}: [Source: {source}]\n{content_preview}")

        # Pass document count info for completeness enforcement
        doc_count = len(candidate_docs)
        expected_ids = ", ".join([f"doc_{i}" for i in range(doc_count)])
        last_doc_idx = doc_count - 1

        prompt = get_prompt(
            "multi_agent_merge_reranking",
            original_question=original_question,
            doc_list="\n\n".join(doc_list),
            doc_count=doc_count,
            expected_ids=expected_ids,
            last_doc_idx=last_doc_idx,
        )

        # Log input candidates
        print(f"\n{'='*60}")
        print(f"LLM RELEVANCE SCORING (Multi-Agent Merge)")
        print(f"Original question: {original_question}")
        print(f"Candidates: {len(candidate_docs)}")
        print(f"\nChunk IDs before scoring:")
        for i, doc in enumerate(candidate_docs):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id}")

        # Prepare fallback scores (position-based if not provided)
        if fallback_scores is None:
            fallback_scores = [100 - (i * 5) for i in range(len(candidate_docs))]

        used_llm_scores = False  # Track if we got valid LLM scores

        try:
            result = self.structured_llm.invoke([HumanMessage(content=prompt)])
            scored_docs = result["scored_documents"]

            # Create score mapping using document_id strings
            score_map = {}
            for doc_score in scored_docs:
                doc_id = doc_score["document_id"]
                if doc_id in doc_id_map:
                    idx = doc_id_map[doc_id]
                    score_map[idx] = doc_score["relevance_score"]
                else:
                    print(f"Warning: Invalid document_id '{doc_id}' in LLM response (expected: {list(doc_id_map.keys())})")

            # If incomplete scoring, fall back to position-based scores
            if len(score_map) != len(candidate_docs):
                missing = [f"doc_{i}" for i in range(len(candidate_docs)) if i not in score_map]
                print(f"Warning: Incomplete LLM scoring ({len(score_map)}/{len(candidate_docs)} docs scored, missing: {missing})")
                print(f"Falling back to position-based ranking")
                scores = fallback_scores
            else:
                # Complete scoring - use LLM scores
                scores = [score_map[i] for i in range(len(candidate_docs))]
                used_llm_scores = True

        except Exception as e:
            print(f"Warning: Reranking failed: {e}. Using fallback scores.")
            scores = fallback_scores

        # Log all scores
        print(f"\nLLM Scores (all {len(candidate_docs)} candidates):")
        for i, (doc, score) in enumerate(zip(candidate_docs, scores)):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id} (score: {score:.1f})")

        # Sort by score and return top_k
        ranked = sorted(
            zip(candidate_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log final selection
        top_k = ranked[:self.top_k]
        print(f"\nFinal selection (top-{self.top_k}):")
        for i, (doc, score) in enumerate(top_k):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id} (score: {score:.1f})")
        print(f"{'='*60}\n")

        top_k_docs = [doc for doc, _ in top_k]
        top_k_scores = [score for _, score in top_k] if used_llm_scores else None
        return top_k_docs, top_k_scores
