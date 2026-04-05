"""Two-stage reranking: CrossEncoder (top-10) then LLM-as-judge (top-4)."""

from typing import List, Tuple
from langchain_core.documents import Document
from .cross_encoder_reranker import CrossEncoderReRanker
from .llm_metadata_reranker import LLMMetadataReRanker


class TwoStageReRanker:
    """Two-stage hybrid reranking: CrossEncoder then LLM-as-judge."""

    def __init__(
        self,
        k_cross_encoder: int = 10,
        k_final: int = 4,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize two-stage hybrid reranker."""
        self.k_cross_encoder = k_cross_encoder
        self.k_final = k_final
        self.cross_encoder = CrossEncoderReRanker(
            model_name=cross_encoder_model,
            top_k=k_cross_encoder
        )
        self.llm_judge = LLMMetadataReRanker(top_k=k_final)

    def rank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Return (document, score) tuples sorted by relevance, limited to k_final."""
        if not documents:
            return []

        # Stage 1: CrossEncoder ranking
        cross_encoder_ranked = self.cross_encoder.rank(query, documents)
        intermediate_docs = [doc for doc, score in cross_encoder_ranked]
        cross_encoder_scores = [score for doc, score in cross_encoder_ranked]

        # Stage 2: LLM reranking with CrossEncoder scores as fallback
        final_ranked = self.llm_judge.rank(query, intermediate_docs, fallback_scores=cross_encoder_scores)

        return final_ranked

    def rank_with_stage_info(
        self,
        query: str,
        documents: List[Document]
    ) -> dict:
        """Rerank with stage info for debugging. Returns {final_ranked, stage1_count, ...}."""
        if not documents:
            return {
                "final_ranked": [],
                "stage1_count": 0,
                "stage2_count": 0,
                "stage1_ranked": []
            }

        # Stage 1: CrossEncoder ranking
        stage1_ranked = self.cross_encoder.rank(query, documents)
        intermediate_docs = [doc for doc, score in stage1_ranked]
        cross_encoder_scores = [score for doc, score in stage1_ranked]

        # Stage 2: LLM reranking with CrossEncoder scores as fallback
        final_ranked = self.llm_judge.rank(query, intermediate_docs, fallback_scores=cross_encoder_scores)

        return {
            "final_ranked": final_ranked,
            "stage1_count": len(stage1_ranked),
            "stage2_count": len(final_ranked),
            "stage1_ranked": stage1_ranked
        }
