"""CrossEncoder-based document reranking using ms-marco-MiniLM-L-6-v2."""

from typing import List, Tuple
from langchain_core.documents import Document


class CrossEncoderReRanker:
    """Rerank documents using CrossEncoder semantic similarity."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        max_length: int = 512,  # MiniLM models trained with 512 token limit
        batch_size: int = 16
    ):
        self.top_k = top_k
        self.batch_size = batch_size

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReRanker.\n"
                "Install with: uv add sentence-transformers"
            )

        self.model = CrossEncoder(model_name, max_length=max_length)

    def rank(
        self,
        query: str,
        documents: List[Document],
        truncate_content: int = 1000  # Allows full table content (<1000 chars) while staying under 512 token limit
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        pairs = [
            [query, doc.page_content[:truncate_content]]
            for doc in documents
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:self.top_k]

    def get_scores_only(
        self,
        query: str,
        documents: List[Document],
        truncate_content: int = 1000
    ) -> List[float]:
        if not documents:
            return []

        pairs = [
            [query, doc.page_content[:truncate_content]]
            for doc in documents
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        return scores.tolist()
