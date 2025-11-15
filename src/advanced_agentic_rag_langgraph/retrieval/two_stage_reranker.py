"""
Hybrid reranking combining CrossEncoder speed with LLM-as-judge quality.

Two-stage reranking pipeline:
1. CrossEncoder: Fast semantic filtering to top-10 documents (~200-300ms)
2. LLM-as-judge: Metadata-aware quality scoring to final top-4 (~300-500ms)

This architecture balances:
- Speed: CrossEncoder pre-filters most documents quickly
- Quality: LLM evaluates only top candidates with full metadata awareness
- Cost: Expensive LLM scoring limited to 10 candidates instead of 20-50
"""

from typing import List, Tuple
from langchain_core.documents import Document
from .cross_encoder_reranker import CrossEncoderReRanker
from .llm_metadata_reranker import LLMMetadataReRanker


class TwoStageReRanker:
    """
    Two-stage hybrid reranking: CrossEncoder then LLM-as-judge.

    Architecture:
        Initial documents (e.g., 20-50)
            |
            v
        Stage 1: CrossEncoder semantic filtering
            produces Top 10 documents (~250ms, ~$0.0001)
            |
            v
        Stage 2: LLM-as-judge with metadata analysis
            produces Final top 4 documents (~400ms, ~$0.005)
            |
            v
        Total: ~650ms, ~$0.006 per query

    Benefits vs. LLM-only reranking:
    - 3-5x faster (650ms vs 2-5s)
    - 5-10x cheaper ($0.006 vs $0.03-0.05)
    - Maintains metadata-aware quality (document type, technical level, domain)
    - Combines semantic similarity (CrossEncoder) with contextual appropriateness (LLM)

    Usage:
        reranker = TwoStageReRanker(k_cross_encoder=10, k_final=4)
        final_docs = reranker.rank("What is attention?", documents)
    """

    def __init__(
        self,
        k_cross_encoder: int = 10,
        k_final: int = 4,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize hybrid reranker with two-stage pipeline.

        Args:
            k_cross_encoder: Number of docs to keep after CrossEncoder stage (default: 10)
            k_final: Number of final docs after LLM stage (default: 4)
            cross_encoder_model: CrossEncoder model name (default: ms-marco-MiniLM-L-6-v2)
            llm_model: LLM model for metadata-aware scoring (default: gpt-4o-mini)
        """
        self.k_cross_encoder = k_cross_encoder
        self.k_final = k_final

        # Stage 1: CrossEncoder for fast semantic filtering
        self.cross_encoder = CrossEncoderReRanker(
            model_name=cross_encoder_model,
            top_k=k_cross_encoder
        )

        # Stage 2: LLM-as-judge for metadata-aware quality scoring
        # Note: LLMMetadataReRanker already uses gpt-4o-mini by default
        self.llm_judge = LLMMetadataReRanker(top_k=k_final)

    def rank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Two-stage reranking: CrossEncoder then LLM-as-judge.

        Args:
            query: User query string
            documents: List of Document objects to rerank

        Returns:
            List of (document, score) tuples from LLM-as-judge,
            sorted by relevance (highest first), limited to k_final

        Example:
            >>> reranker = TwoStageReRanker(k_cross_encoder=10, k_final=4)
            >>> docs = retriever.get_relevant_documents("attention mechanism", k=20)
            >>> final_docs = reranker.rank("attention mechanism", docs)
            >>> # Returns 4 best documents after two-stage filtering
        """
        if not documents:
            return []

        # Stage 1: CrossEncoder filtering
        # Fast semantic similarity to reduce candidate set
        cross_encoder_ranked = self.cross_encoder.rank(query, documents)

        # Extract just the documents (discard CrossEncoder scores)
        # These scores are replaced by LLM scores in stage 2
        intermediate_docs = [doc for doc, score in cross_encoder_ranked]

        # Stage 2: LLM-as-judge quality scoring
        # Metadata-aware ranking (document type, technical level, domain)
        # Returns (doc, score) tuples with LLM relevance scores
        final_ranked = self.llm_judge.rank(query, intermediate_docs)

        return final_ranked

    def rank_with_stage_info(
        self,
        query: str,
        documents: List[Document]
    ) -> dict:
        """
        Two-stage reranking with detailed stage information.

        Useful for debugging and analysis of reranking decisions.

        Args:
            query: User query string
            documents: List of Document objects to rerank

        Returns:
            Dictionary with:
            - final_ranked: List of (document, llm_score) tuples
            - stage1_count: Number of docs after CrossEncoder
            - stage2_count: Number of final docs
            - stage1_ranked: Full CrossEncoder results with scores
        """
        if not documents:
            return {
                "final_ranked": [],
                "stage1_count": 0,
                "stage2_count": 0,
                "stage1_ranked": []
            }

        # Stage 1: CrossEncoder
        stage1_ranked = self.cross_encoder.rank(query, documents)
        intermediate_docs = [doc for doc, score in stage1_ranked]

        # Stage 2: LLM-as-judge
        final_ranked = self.llm_judge.rank(query, intermediate_docs)

        return {
            "final_ranked": final_ranked,
            "stage1_count": len(stage1_ranked),
            "stage2_count": len(final_ranked),
            "stage1_ranked": stage1_ranked
        }
