"""
CrossEncoder-based document reranking for improved retrieval quality.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for fast, accurate relevance scoring.
Processes query-document pairs jointly to capture semantic relationships.
"""

from typing import List, Tuple
from langchain_core.documents import Document


class CrossEncoderReRanker:
    """
    Rerank documents using CrossEncoder semantic similarity.

    Performance: 20-35% accuracy improvement, 200-500ms latency, ~100x cost reduction vs LLM.
    Model: ms-marco-MiniLM-L-6-v2 (200-300ms for 10-20 docs).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        max_length: int = 512,  # MiniLM models trained with 512 token limit
        batch_size: int = 16
    ):
        import os

        self.top_k = top_k
        self.batch_size = batch_size

        # Check for mock mode (allows running tests without model downloads)
        self.use_mock = os.getenv("USE_MOCK_MODELS", "false").lower() == "true"

        if self.use_mock:
            print("[MOCK MODE] CrossEncoder using placeholder scores (no model download)")
            self.model = None
        else:
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
        truncate_content: int = 400  # Prevents exceeding 512 token limit
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        # Mock mode: Return fake decreasing scores
        if self.use_mock:
            import numpy as np
            num_docs = len(documents)
            # Generate fake scores with slight variation (0.9, 0.85, 0.8, ...)
            scores = np.linspace(0.9, 0.5, num_docs)
            ranked = list(zip(documents, scores))
            return ranked[:self.top_k]

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
        truncate_content: int = 400
    ) -> List[float]:
        if not documents:
            return []

        # Mock mode: Return fake scores
        if self.use_mock:
            import numpy as np
            num_docs = len(documents)
            scores = np.linspace(0.9, 0.5, num_docs)
            return scores.tolist()

        pairs = [
            [query, doc.page_content[:truncate_content]]
            for doc in documents
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        return scores.tolist()
